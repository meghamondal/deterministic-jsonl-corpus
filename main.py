import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set

import google_crc32c
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# --- Helper Utilities & Regexes ---

# URI: gs://bucket/object (bucket and object path must be non-empty)
GS_URI_REGEX = re.compile(r"^gs://[^/]+/.+$")

# CRC32C: exactly 8 lowercase hex digits
CRC32C_REGEX = re.compile(r"^[0-9a-f]{8}$")

# Decimal string for generations
DECIMAL_REGEX = re.compile(r"^[0-9]+$")

# ISO-8601 Timestamp Regex: YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
ISO_TIMESTAMP_REGEX = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|([+-])(\d{2}):(\d{2}))$"
)

# JS Safe Integer Limit (2^53 - 1)
JS_MAX_SAFE_INTEGER = 9007199254740991


def compute_crc32c_hex(content_bytes: bytes) -> str:
    """Computes CRC32C over exact UTF-8 bytes and returns 8-char lowercase hex string."""
    checksum = google_crc32c.Checksum(content_bytes)
    return checksum.digest().hex().lower()


def parse_iso8601(time_str: Any) -> Optional[datetime]:
    """Validates calendar, time offset rules, and normalizes to UTC datetime."""
    if not isinstance(time_str, str):
        return None

    match = ISO_TIMESTAMP_REGEX.match(time_str)
    if not match:
        return None

    year, month, day, hour, minute, second, frac, tz_str, sign, tz_h, tz_m = match.groups()

    y, m, d = int(year), int(month), int(day)
    hh, mm, ss = int(hour), int(minute), int(second)

    # Microsecond fraction parsing (1 to 3 digits)
    if frac:
        ms = int(frac.ljust(3, "0")) * 1000
    else:
        ms = 0

    # Handle timezone offset validation
    if tz_str == "Z":
        tz_offset = timedelta(0)
    else:
        th, tm = int(tz_h), int(tz_m)
        if th > 14 or tm > 59:
            return None
        if th == 14 and tm != 0:
            return None
        
        offset_mins = th * 60 + tm
        if sign == "-":
            offset_mins = -offset_mins
        tz_offset = timedelta(minutes=offset_mins)

    # Validate actual calendar dates & time
    try:
        dt = datetime(y, m, d, hh, mm, ss, ms, tzinfo=timezone(tz_offset))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def format_iso8601_utc(dt: datetime) -> str:
    """Formats a UTC datetime to YYYY-MM-DDTHH:mm:ss.sssZ."""
    ms = dt.microsecond // 1000
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}Z"


def canonicalize_text(text: str) -> str:
    """Unicode NFKC, lowercase, trim, and collapse Unicode whitespace to one ASCII space."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    words = text.split()
    return " ".join(words)


def extract_words(text: str) -> Set[str]:
    """Extracts set of lowercase Unicode letter/number words for Jaccard calculation."""
    words = set()
    current_word = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("L") or cat.startswith("N"):
            current_word.append(char)
        else:
            if current_word:
                words.add("".join(current_word))
                current_word = []
    if current_word:
        words.add("".join(current_word))
    return words


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity. Empty/empty similarity is 1.0."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union


def json_compact(obj: Any) -> str:
    """Returns compact JSON string without extra spacing."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# --- Endpoint Definition ---

@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if "policy" not in body or "objects" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    policy = body.get("policy")
    objects = body.get("objects")

    if not isinstance(policy, dict) or not isinstance(objects, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # --- Policy Validation ---
    policy_valid = True
    min_dt, max_dt = None, None
    thresh = None

    if "minTime" not in policy or "maxTime" not in policy or "contaminationThreshold" not in policy:
        policy_valid = False
    else:
        min_dt = parse_iso8601(policy.get("minTime"))
        max_dt = parse_iso8601(policy.get("maxTime"))
        thresh = policy.get("contaminationThreshold")

        if min_dt is None or max_dt is None or min_dt > max_dt:
            policy_valid = False

        if not isinstance(thresh, (int, float)) or isinstance(thresh, bool):
            policy_valid = False
        elif thresh < 0.0 or thresh > 1.0:
            policy_valid = False

    # --- Processing Objects & Identity/Integrity Checks ---

    accepted_objects = []
    rejected_objects = []
    lineage = []

    for obj in objects:
        if not isinstance(obj, dict):
            rejected_objects.append({"uri": None, "reasonCodes": ["SCHEMA_INVALID"]})
            continue

        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_gen = obj.get("fetchedGeneration")
        crc32c = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

        reasons = set()

        # 1. URI_INVALID (Must be string and match gs://bucket/object)
        supplied_uri = uri if isinstance(uri, str) else None
        if not isinstance(uri, str) or not GS_URI_REGEX.match(uri):
            reasons.add("URI_INVALID")

        # 2. GENERATION_INVALID & GENERATION_MISMATCH
        gen_valid = isinstance(generation, str) and bool(DECIMAL_REGEX.match(generation))
        fetched_gen_valid = isinstance(fetched_gen, str) and bool(DECIMAL_REGEX.match(fetched_gen))

        if not gen_valid or not fetched_gen_valid:
            reasons.add("GENERATION_INVALID")

        # GENERATION_MISMATCH applies when both are valid OR when they are both strings that differ
        if isinstance(generation, str) and isinstance(fetched_gen, str) and generation != fetched_gen:
            reasons.add("GENERATION_MISMATCH")

        # 3. CRC32C_INVALID & CRC32C_MISMATCH
        crc_syntax_valid = isinstance(crc32c, str) and bool(CRC32C_REGEX.match(crc32c))
        if not crc_syntax_valid:
            reasons.add("CRC32C_INVALID")
        elif isinstance(content, str):
            # Check mismatch ONLY if content is string and CRC syntax is valid
            computed_crc = compute_crc32c_hex(content.encode("utf-8"))
            if computed_crc != crc32c:
                reasons.add("CRC32C_MISMATCH")

        # 4. SCHEMA_INVALID & JSONL_INVALID
        if schema_id != "training-v1" or not isinstance(content, str):
            reasons.add("SCHEMA_INVALID")

        parsed_rows = []
        if isinstance(content, str):
            lines = content.split("\n")
            non_blank_lines = [line for line in lines if line.strip() != ""]

            if len(non_blank_lines) == 0:
                reasons.add("SCHEMA_INVALID")  # Empty content / no valid rows
            else:
                json_parse_failed = False
                row_schema_failed = False

                for line in non_blank_lines:
                    try:
                        row_data = json.loads(line)
                    except Exception:
                        json_parse_failed = True
                        break

                    if not isinstance(row_data, dict):
                        row_schema_failed = True
                        break

                    # Must contain EXACTLY the 5 keys: id, entity, eventTime, revision, text
                    keys = set(row_data.keys())
                    expected_keys = {"id", "entity", "eventTime", "revision", "text"}
                    if keys != expected_keys:
                        row_schema_failed = True
                        break

                    r_id = row_data["id"]
                    r_ent = row_data["entity"]
                    r_time = row_data["eventTime"]
                    r_rev = row_data["revision"]
                    r_text = row_data["text"]

                    if not (
                        isinstance(r_id, str)
                        and isinstance(r_ent, str)
                        and isinstance(r_time, str)
                        and isinstance(r_text, str)
                    ):
                        row_schema_failed = True
                        break

                    # Revision must be non-negative safe integer (exclude booleans!)
                    if (
                        type(r_rev) is not int
                        or isinstance(r_rev, bool)
                        or r_rev < 0
                        or r_rev > JS_MAX_SAFE_INTEGER
                    ):
                        row_schema_failed = True
                        break

                    # Validate eventTime format
                    parsed_time = parse_iso8601(r_time)
                    if parsed_time is None:
                        row_schema_failed = True
                        break

                    parsed_rows.append(
                        {
                            "id": r_id,
                            "entity": r_ent,
                            "eventTime_dt": parsed_time,
                            "revision": r_rev,
                            "text": r_text,
                        }
                    )

                if json_parse_failed:
                    reasons.add("JSONL_INVALID")
                elif row_schema_failed:
                    reasons.add("SCHEMA_INVALID")

        if reasons:
            reasons_sorted = sorted(list(reasons), key=lambda x: x.encode("utf-8"))
            rejected_objects.append({"uri": supplied_uri, "reasonCodes": reasons_sorted})
        else:
            accepted_objects.append(
                {
                    "uri": uri,
                    "generation": generation,
                    "crc32c": crc32c,
                    "schemaId": schema_id,
                    "rows": parsed_rows,
                }
            )
            lineage.append(
                {
                    "uri": uri,
                    "generation": generation,
                    "crc32c": crc32c,
                    "schemaId": schema_id,
                }
            )

    # --- Processing Rows: Canonicalization & Deduplication ---

    retained_rows_map = {}
    all_retained_candidates = []

    for obj in accepted_objects:
        for r in obj["rows"]:
            c_entity = canonicalize_text(r["entity"])
            c_text = canonicalize_text(r["text"])
            c_eventTime = format_iso8601_utc(r["eventTime_dt"])

            canon_row = {
                "id": r["id"],
                "entity": c_entity,
                "eventTime": c_eventTime,
                "eventTime_dt": r["eventTime_dt"],
                "revision": r["revision"],
                "text": c_text,
            }

            key = (c_entity, c_eventTime, c_text)

            if key not in retained_rows_map:
                retained_rows_map[key] = canon_row
            else:
                existing = retained_rows_map[key]
                existing_id_bytes = existing["id"].encode("utf-8")
                new_id_bytes = canon_row["id"].encode("utf-8")

                if canon_row["revision"] > existing["revision"]:
                    retained_rows_map[key] = canon_row
                elif canon_row["revision"] == existing["revision"]:
                    if new_id_bytes < existing_id_bytes:
                        retained_rows_map[key] = canon_row

            all_retained_candidates.append(canon_row)

    winning_rows = list(retained_rows_map.values())
    winning_ids = {id(r) for r in winning_rows}

    rejected_rows_dict: Dict[str, Set[str]] = {}

    def add_row_rejection(row_id: str, code: str):
        if row_id not in rejected_rows_dict:
            rejected_rows_dict[row_id] = set()
        rejected_rows_dict[row_id].add(code)

    for cand in all_retained_candidates:
        if id(cand) not in winning_ids:
            add_row_rejection(cand["id"], "DUPLICATE")

    # --- Policy & Window Validation for Retained Rows ---

    valid_retained_rows = []

    for r in winning_rows:
        row_id = r["id"]
        if not policy_valid:
            add_row_rejection(row_id, "POLICY_INVALID")
        else:
            if r["eventTime_dt"] < min_dt or r["eventTime_dt"] > max_dt:
                add_row_rejection(row_id, "OUT_OF_WINDOW")
            else:
                valid_retained_rows.append(r)

    # --- Split Assignment & Contamination Filtering ---

    train_rows = []
    val_rows = []
    test_rows = []

    for r in valid_retained_rows:
        entity_bytes = r["entity"].encode("utf-8")
        sha = hashlib.sha256(entity_bytes).digest()
        first_byte = sha[0]
        bucket = first_byte % 10

        if 0 <= bucket <= 5:
            train_rows.append(r)
        elif 6 <= bucket <= 7:
            val_rows.append(r)
        else:
            test_rows.append(r)

    train_word_sets = [extract_words(tr["text"]) for tr in train_rows]

    final_train = train_rows
    final_val = []
    final_test = []

    def check_contamination(row: dict) -> bool:
        row_words = extract_words(row["text"])
        for tw_set in train_word_sets:
            if jaccard_similarity(row_words, tw_set) >= thresh:
                return True
        return False

    for r in val_rows:
        if check_contamination(r):
            add_row_rejection(r["id"], "TRAIN_CONTAMINATION")
        else:
            final_val.append(r)

    for r in test_rows:
        if check_contamination(r):
            add_row_rejection(r["id"], "TRAIN_CONTAMINATION")
        else:
            final_test.append(r)

    # --- Sorting and Serialization ---

    def row_sort_key(r: dict) -> Tuple[bytes, str]:
        compact_dict = {
            "id": r["id"],
            "entity": r["entity"],
            "eventTime": r["eventTime"],
            "revision": r["revision"],
            "text": r["text"],
        }
        return (r["id"].encode("utf-8"), json_compact(compact_dict))

    def process_split(split_rows: list) -> Tuple[list, str]:
        sorted_rows = sorted(split_rows, key=row_sort_key)
        serialized_rows = []
        bytes_buffer = bytearray()

        for r in sorted_rows:
            compact_obj = {
                "id": r["id"],
                "entity": r["entity"],
                "eventTime": r["eventTime"],
                "revision": r["revision"],
                "text": r["text"],
            }
            serialized_rows.append(compact_obj)
            line_str = json_compact(compact_obj) + "\n"
            bytes_buffer.extend(line_str.encode("utf-8"))

        digest = hashlib.sha256(bytes_buffer).hexdigest()
        return serialized_rows, digest

    train_serialized, train_digest = process_split(final_train)
    val_serialized, val_digest = process_split(final_val)
    test_serialized, test_digest = process_split(final_test)

    # --- Formatting Rejected Objects, Rejected Rows, and Lineage ---

    def rejected_obj_sort_key(obj: dict) -> Tuple[bytes, str]:
        uri_bytes = obj["uri"].encode("utf-8") if isinstance(obj["uri"], str) else b""
        return (uri_bytes, json_compact(obj))

    sorted_rejected_objects = sorted(rejected_objects, key=rejected_obj_sort_key)

    formatted_rejected_rows = []
    for r_id, codes in rejected_rows_dict.items():
        codes_sorted = sorted(list(codes), key=lambda x: x.encode("utf-8"))
        formatted_rejected_rows.append({"id": r_id, "reasonCodes": codes_sorted})

    def rejected_row_sort_key(r: dict) -> Tuple[bytes, str]:
        return (r["id"].encode("utf-8"), json_compact(r))

    sorted_rejected_rows = sorted(formatted_rejected_rows, key=rejected_row_sort_key)

    def lineage_sort_key(lin: dict) -> Tuple[bytes, str]:
        return (lin["uri"].encode("utf-8"), json_compact(lin))

    sorted_lineage = sorted(lineage, key=lineage_sort_key)

    response_data = {
        "splits": {
            "train": train_serialized,
            "validation": val_serialized,
            "test": test_serialized,
        },
        "rejectedObjects": sorted_rejected_objects,
        "rejectedRows": sorted_rejected_rows,
        "digests": {
            "train": train_digest,
            "validation": val_digest,
            "test": test_digest,
        },
        "lineage": sorted_lineage,
    }

    return JSONResponse(status_code=200, content=response_data)
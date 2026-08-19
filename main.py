from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import hashlib
import unicodedata
import re
from datetime import datetime, timezone
from math import isfinite

app = FastAPI()


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}

MAX_SAFE_INTEGER = 9007199254740991

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

GENERATION_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

OBJ_CODES = {
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
}

ROW_CODES = {
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
}


# ============================================================
# CRC32C
# ============================================================

POLY = 0x82F63B78
CRC_TABLE = []

for i in range(256):
    c = i

    for _ in range(8):
        if c & 1:
            c = (c >> 1) ^ POLY
        else:
            c >>= 1

    CRC_TABLE.append(c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF

    for byte in data:
        crc = CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    return (~crc) & 0xFFFFFFFF


# ============================================================
# DETERMINISTIC HELPERS
# ============================================================

def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def compact_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sorted_reason_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: x.encode("utf-8")
    )


# ============================================================
# URI
# ============================================================

def valid_uri(uri) -> bool:
    if not isinstance(uri, str):
        return False

    # gs://bucket/object
    # Bucket must be non-empty and object must be non-empty.
    return re.fullmatch(r"gs://[^/]+/.+", uri) is not None


# ============================================================
# GENERATION
# ============================================================

def valid_generation(value) -> bool:
    return (
        isinstance(value, str)
        and GENERATION_RE.fullmatch(value) is not None
    )


# ============================================================
# TIME
# ============================================================

def parse_time(value):
    """
    Accept:
      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sZ
      YYYY-MM-DDTHH:mm:ss.ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ

    or the same with ±HH:mm offset.

    Calendar validity is delegated to datetime.
    Offset magnitude is explicitly checked.
    """

    if not isinstance(value, str):
        return None

    if TIME_RE.fullmatch(value) is None:
        return None

    try:
        if value.endswith("Z"):
            iso_value = value[:-1] + "+00:00"
        else:
            iso_value = value

        dt = datetime.fromisoformat(iso_value)

        offset = dt.utcoffset()

        if offset is None:
            return None

        total_seconds = int(offset.total_seconds())

        absolute_minutes = abs(total_seconds) // 60

        offset_hours = absolute_minutes // 60
        offset_minutes = absolute_minutes % 60

        # Maximum magnitude is 14:00.
        # Therefore 14:01, 14:30 etc. are invalid.
        if offset_hours > 14:
            return None

        if offset_hours == 14 and offset_minutes != 0:
            return None

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def normalize_event_time(dt) -> str:
    milliseconds = dt.microsecond // 1000

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{milliseconds:03d}Z"
    )


# ============================================================
# CANONICALIZATION
# ============================================================

def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)

    value = value.lower()

    value = value.strip()

    # Unicode whitespace -> one ASCII space.
    value = re.sub(r"\s+", " ", value)

    return value


# ============================================================
# ROW VALIDATION
# ============================================================

def valid_row(row) -> bool:
    if not isinstance(row, dict):
        return False

    # Exactly the required five keys.
    if set(row.keys()) != EXPECTED_ROW_KEYS:
        return False

    # Four text fields.
    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not isinstance(row["eventTime"], str):
        return False

    if not isinstance(row["text"], str):
        return False

    # Revision: non-negative safe integer.
    bool_is_int = isinstance(row["revision"], bool)

    if (
        not isinstance(row["revision"], int)
        or bool_is_int
        or row["revision"] < 0
        or row["revision"] > MAX_SAFE_INTEGER
    ):
        return False

    # Event time must be valid.
    if parse_time(row["eventTime"]) is None:
        return False

    return True


# ============================================================
# WORD SET / JACCARD
# ============================================================

def word_set(text: str):
    """
    Lowercase Unicode letter/number word set.

    A word consists of consecutive Unicode letters/numbers.
    """

    words = re.findall(r"[\w]+", text, flags=re.UNICODE)

    result = set()

    for word in words:
        cleaned = "".join(
            ch for ch in word
            if ch.isalnum()
        )

        if cleaned:
            result.add(cleaned.lower())

    return result


def jaccard(a, b) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# REQUEST
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    # --------------------------------------------------------
    # Parse JSON request
    # --------------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    # Missing policy -> INVALID_INPUT.
    #
    # A policy that exists but is malformed is NOT an invalid
    # request. It makes the policy invalid and therefore all
    # retained rows get POLICY_INVALID.
    if "policy" not in body:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    if "objects" not in body or not isinstance(body["objects"], list):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    policy = body["policy"]
    objects = body["objects"]

    rejected_objects = []
    rejected_rows = []
    lineage = []

    parsed_rows = []


    # ========================================================
    # OBJECT PROCESSING
    # ========================================================

    for obj in objects:

        # ----------------------------------------------------
        # Object must be an object.
        # Since all independent object checks are applicable,
        # a non-dict object gets the relevant object-level
        # failures.
        # ----------------------------------------------------

        if not isinstance(obj, dict):

            rejected_objects.append({
                "uri": None,
                "reasonCodes": sorted_reason_codes([
                    "URI_INVALID",
                    "GENERATION_INVALID",
                    "CRC32C_INVALID",
                    "SCHEMA_INVALID",
                ])
            })

            continue


        # ----------------------------------------------------
        # URI
        # ----------------------------------------------------

        uri = obj.get("uri")

        reasons = []

        if not valid_uri(uri):
            reasons.append("URI_INVALID")


        # ----------------------------------------------------
        # GENERATIONS
        # ----------------------------------------------------

        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")

        generation_valid = valid_generation(generation)
        fetched_generation_valid = valid_generation(
            fetched_generation
        )

        if not generation_valid or not fetched_generation_valid:
            reasons.append("GENERATION_INVALID")

        # The supplied values are unequal.
        #
        # We check this independently from validity so that
        # every applicable object code is emitted.
        if generation != fetched_generation:
            reasons.append("GENERATION_MISMATCH")


        # ----------------------------------------------------
        # CRC32C
        # ----------------------------------------------------

        crc = obj.get("crc32c")

        crc_valid = (
            isinstance(crc, str)
            and CRC_RE.fullmatch(crc) is not None
        )

        if not crc_valid:
            reasons.append("CRC32C_INVALID")


        # ----------------------------------------------------
        # SCHEMA / CONTENT
        # ----------------------------------------------------

        content = obj.get("content")
        schema_id = obj.get("schemaId")

        if (
            not isinstance(content, str)
            or schema_id != "training-v1"
        ):
            reasons.append("SCHEMA_INVALID")


        # ----------------------------------------------------
        # CRC32C MISMATCH
        #
        # Only check mismatch when:
        #   - content is a string
        #   - CRC syntax is valid
        # ----------------------------------------------------

        if isinstance(content, str) and crc_valid:

            actual_crc = f"{crc32c(content.encode('utf-8')):08x}"

            if actual_crc != crc:
                reasons.append("CRC32C_MISMATCH")


        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        rows = []
        jsonl_invalid = False

        if isinstance(content, str):

            for line in content.splitlines():

                # Blank lines ignored.
                if not line.strip():
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    jsonl_invalid = True
                    break

                rows.append(parsed)

            if jsonl_invalid:
                reasons.append("JSONL_INVALID")

            # Every file must contain at least one row.
            if not rows and not jsonl_invalid:
                reasons.append("SCHEMA_INVALID")


        # ----------------------------------------------------
        # ROW SCHEMA
        #
        # We validate all successfully parsed rows even if
        # another object-level error already exists.
        # ----------------------------------------------------

        valid_rows = []

        for row in rows:

            if not valid_row(row):
                reasons.append("SCHEMA_INVALID")
                continue

            valid_rows.append(row)


        # ----------------------------------------------------
        # OBJECT REJECTION
        # ----------------------------------------------------

        if reasons:

            rejected_objects.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sorted_reason_codes(reasons)
            })

            continue


        # ----------------------------------------------------
        # ACCEPTED OBJECT -> LINEAGE
        # ----------------------------------------------------

        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": crc,
            "schemaId": schema_id
        })


        # ----------------------------------------------------
        # CANONICALIZE ACCEPTED ROWS
        # ----------------------------------------------------

        for row in valid_rows:

            event_dt = parse_time(row["eventTime"])

            parsed_rows.append({
                "id": row["id"],
                "entity": canonical_text(row["entity"]),
                "eventTime": normalize_event_time(event_dt),
                "revision": row["revision"],
                "text": canonical_text(row["text"])
            })


    # ========================================================
    # DEDUPLICATION
    # ========================================================

    groups = {}

    for row in parsed_rows:

        dedup_key = (
            row["entity"],
            row["eventTime"],
            row["text"]
        )

        groups.setdefault(dedup_key, []).append(row)


    retained = []

    for rows in groups.values():

        # Highest revision first.
        #
        # If revisions tie, UTF-8-byte-smallest ID wins.
        rows.sort(
            key=lambda row: (
                -row["revision"],
                utf8_key(row["id"])
            )
        )

        winner = rows[0]

        retained.append(winner)

        # Every other row is a duplicate loser.
        for loser in rows[1:]:

            rejected_rows.append({
                "id": loser["id"],
                "reasonCodes": ["DUPLICATE"]
            })


    # ========================================================
    # POLICY
    # ========================================================

    policy_valid = False

    min_dt = None
    max_dt = None
    threshold = None

    if isinstance(policy, dict):

        min_dt = parse_time(policy.get("minTime"))
        max_dt = parse_time(policy.get("maxTime"))

        threshold = policy.get(
            "contaminationThreshold"
        )

        if (
            min_dt is not None
            and max_dt is not None
            and min_dt <= max_dt
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and isfinite(threshold)
            and 0 <= threshold <= 1
        ):
            policy_valid = True


    candidates = []

    for row in retained:

        if not policy_valid:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": ["POLICY_INVALID"]
            })

            continue


        row_dt = parse_time(row["eventTime"])

        if row_dt < min_dt or row_dt > max_dt:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": ["OUT_OF_WINDOW"]
            })

            continue


        candidates.append(row)


    # ========================================================
    # SPLIT
    # ========================================================

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }


    for row in candidates:

        first_byte = hashlib.sha256(
            row["entity"].encode("utf-8")
        ).digest()[0]

        bucket = first_byte % 10

        if 0 <= bucket <= 5:
            splits["train"].append(row)

        elif 6 <= bucket <= 7:
            splits["validation"].append(row)

        else:
            splits["test"].append(row)


    # ========================================================
    # TRAIN WORD SETS
    # ========================================================

    train_word_sets = [
        word_set(row["text"])
        for row in splits["train"]
    ]


    # ========================================================
    # CONTAMINATION
    # ========================================================

    for split_name in ("validation", "test"):

        kept = []

        for row in splits[split_name]:

            current_words = word_set(row["text"])

            contaminated = False

            for train_words in train_word_sets:

                similarity = jaccard(
                    current_words,
                    train_words
                )

                if similarity >= threshold:
                    contaminated = True
                    break


            if contaminated:

                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": [
                        "TRAIN_CONTAMINATION"
                    ]
                })

            else:
                kept.append(row)


        splits[split_name] = kept


    # ========================================================
    # DETERMINISTIC ARTIFACTS
    # ========================================================

    digests = {}

    for split_name in (
        "train",
        "validation",
        "test"
    ):

        # Primary key: UTF-8 bytes of ID.
        #
        # Secondary key: compact row JSON.
        splits[split_name].sort(
            key=lambda row: (
                utf8_key(row["id"]),
                compact_json(row).encode("utf-8")
            )
        )


        # Exact JSONL bytes.
        payload = b""

        for row in splits[split_name]:

            line = compact_json(row) + "\n"

            payload += line.encode("utf-8")


        digests[split_name] = hashlib.sha256(
            payload
        ).hexdigest()


    # ========================================================
    # REJECTED OBJECTS SORT
    # ========================================================

    rejected_objects.sort(
        key=lambda item: (
            (
                utf8_key(item["uri"])
                if isinstance(item["uri"], str)
                else b""
            ),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # REJECTED ROWS SORT
    # ========================================================

    rejected_rows.sort(
        key=lambda item: (
            utf8_key(item["id"]),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # LINEAGE SORT
    # ========================================================

    lineage.sort(
        key=lambda item: (
            utf8_key(item["uri"]),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "splits": {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"]
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": digests["train"],
            "validation": digests["validation"],
            "test": digests["test"]
        },
        "lineage": lineage
    }
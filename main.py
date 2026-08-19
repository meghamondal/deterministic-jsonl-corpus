from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from math import isfinite

app = FastAPI()

MAX_SAFE_INTEGER = 9007199254740991

EXPECTED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

GENERATION_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")


# ============================================================
# CRC32C
# ============================================================

CRC32C_POLY = 0x82F63B78

CRC32C_TABLE = []

for i in range(256):
    crc = i

    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ CRC32C_POLY
        else:
            crc >>= 1

    CRC32C_TABLE.append(crc)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF

    for byte in data:
        crc = (
            CRC32C_TABLE[(crc ^ byte) & 0xFF]
            ^ (crc >> 8)
        )

    return (~crc) & 0xFFFFFFFF


# ============================================================
# JSON / SORT HELPERS
# ============================================================

def compact_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def sort_reason_codes(codes):
    return sorted(
        set(codes),
        key=lambda x: utf8(x)
    )


# ============================================================
# URI
# ============================================================

def valid_uri(value) -> bool:
    if not isinstance(value, str):
        return False

    # gs://bucket/object
    return re.fullmatch(
        r"gs://[^/]+/.+",
        value
    ) is not None


# ============================================================
# GENERATION
# ============================================================

def valid_generation(value) -> bool:
    if not isinstance(value, str):
        return False

    return GENERATION_RE.fullmatch(value) is not None


# ============================================================
# TIME
# ============================================================

def parse_time(value):
    if not isinstance(value, str):
        return None

    if TIME_RE.fullmatch(value) is None:
        return None

    try:
        if value.endswith("Z"):
            iso = value[:-1] + "+00:00"
        else:
            iso = value

        dt = datetime.fromisoformat(iso)

        offset = dt.utcoffset()

        if offset is None:
            return None

        total_minutes = abs(
            int(offset.total_seconds()) // 60
        )

        hours = total_minutes // 60
        minutes = total_minutes % 60

        # Maximum allowed offset is 14:00.
        if hours > 14:
            return None

        if hours == 14 and minutes != 0:
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

def canonicalize(value: str) -> str:
    # Unicode NFKC
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Trim
    value = value.strip()

    # Collapse Unicode whitespace to one ASCII space
    value = re.sub(r"\s+", " ", value)

    return value


# ============================================================
# ROW VALIDATION
# ============================================================

def valid_row(row) -> bool:

    if not isinstance(row, dict):
        return False

    # Exactly these five keys.
    if set(row.keys()) != EXPECTED_ROW_KEYS:
        return False

    # id, entity, eventTime, text must be strings.
    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not isinstance(row["eventTime"], str):
        return False

    if not isinstance(row["text"], str):
        return False

    # revision must be a non-negative safe integer.
    if isinstance(row["revision"], bool):
        return False

    if not isinstance(row["revision"], int):
        return False

    if row["revision"] < 0:
        return False

    if row["revision"] > MAX_SAFE_INTEGER:
        return False

    # eventTime must be valid.
    if parse_time(row["eventTime"]) is None:
        return False

    return True


# ============================================================
# WORD SET / JACCARD
# ============================================================

def word_set(text: str):
    """
    Lowercase Unicode letter/number word-set.

    Unicode letters/numbers are retained.
    Punctuation and separators delimit words.
    """

    words = set()
    current = []

    for char in text:

        if char.isalnum():
            current.append(char)
        else:
            if current:
                words.add(
                    "".join(current).lower()
                )
                current = []

    if current:
        words.add(
            "".join(current).lower()
        )

    return words


def jaccard(a, b) -> float:

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    # --------------------------------------------------------
    # REQUEST PARSING
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

    # Missing policy -> HTTP 400.
    if "policy" not in body:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    # objects must exist and be an array.
    if (
        "objects" not in body
        or not isinstance(body["objects"], list)
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400
        )

    policy = body["policy"]
    objects = body["objects"]

    rejected_objects = []
    rejected_rows = []
    lineage = []

    accepted_rows = []


    # ========================================================
    # OBJECT PROCESSING
    # ========================================================

    for obj in objects:

        # An object entry which isn't an object cannot provide
        # valid object fields.
        if not isinstance(obj, dict):
            rejected_objects.append({
                "uri": None,
                "reasonCodes": [
                    "SCHEMA_INVALID"
                ]
            })
            continue

        reasons = []

        # ----------------------------------------------------
        # URI
        # ----------------------------------------------------

        uri = obj.get("uri")

        if not valid_uri(uri):
            reasons.append("URI_INVALID")


        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        generation = obj.get("generation")
        fetched_generation = obj.get(
            "fetchedGeneration"
        )

        generation_valid = valid_generation(
            generation
        )

        fetched_generation_valid = valid_generation(
            fetched_generation
        )

        if (
            not generation_valid
            or not fetched_generation_valid
        ):
            reasons.append("GENERATION_INVALID")

        # Equality is checked independently.
        if generation != fetched_generation:
            reasons.append("GENERATION_MISMATCH")


        # ----------------------------------------------------
        # CRC32C SYNTAX
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

        if not isinstance(content, str):
            reasons.append("SCHEMA_INVALID")

        if schema_id != "training-v1":
            reasons.append("SCHEMA_INVALID")


        # ----------------------------------------------------
        # CRC32C VALUE
        #
        # Only compare when:
        #   content is a string
        #   CRC syntax is valid
        # ----------------------------------------------------

        if (
            isinstance(content, str)
            and crc_valid
        ):
            actual_crc = (
                f"{crc32c(content.encode('utf-8')):08x}"
            )

            if actual_crc != crc:
                reasons.append("CRC32C_MISMATCH")


        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        rows = []
        jsonl_invalid = False

        if isinstance(content, str):

            for line in content.splitlines():

                # Blank lines are ignored.
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

            # No non-blank rows.
            elif not rows:
                reasons.append("SCHEMA_INVALID")


        # ----------------------------------------------------
        # ROW SCHEMA
        # ----------------------------------------------------

        valid_rows = []

        for row in rows:

            if not valid_row(row):
                reasons.append("SCHEMA_INVALID")
                continue

            valid_rows.append(row)


        # ----------------------------------------------------
        # OBJECT RESULT
        # ----------------------------------------------------

        reasons = sort_reason_codes(reasons)

        if reasons:

            rejected_objects.append({
                "uri": (
                    uri
                    if isinstance(uri, str)
                    else None
                ),
                "reasonCodes": reasons
            })

            continue


        # ----------------------------------------------------
        # LINEAGE
        # ----------------------------------------------------

        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": crc,
            "schemaId": schema_id
        })


        # ----------------------------------------------------
        # CANONICALIZE ROWS
        # ----------------------------------------------------

        for row in valid_rows:

            event_dt = parse_time(
                row["eventTime"]
            )

            accepted_rows.append({
                "id": row["id"],
                "entity": canonicalize(
                    row["entity"]
                ),
                "eventTime": normalize_event_time(
                    event_dt
                ),
                "revision": row["revision"],
                "text": canonicalize(
                    row["text"]
                )
            })


    # ========================================================
    # DEDUPLICATION
    # ========================================================

    groups = {}

    for row in accepted_rows:

        key = (
            row["entity"],
            row["eventTime"],
            row["text"]
        )

        groups.setdefault(
            key,
            []
        ).append(row)


    retained_rows = []

    for rows in groups.values():

        # Highest revision wins.
        #
        # On equal revision:
        # smallest UTF-8 ID wins.
        rows.sort(
            key=lambda row: (
                -row["revision"],
                utf8(row["id"])
            )
        )

        winner = rows[0]

        retained_rows.append(winner)

        for loser in rows[1:]:

            rejected_rows.append({
                "id": loser["id"],
                "reasonCodes": [
                    "DUPLICATE"
                ]
            })


    # ========================================================
    # POLICY
    # ========================================================

    policy_valid = False

    min_time = None
    max_time = None
    contamination_threshold = None

    if isinstance(policy, dict):

        min_time = parse_time(
            policy.get("minTime")
        )

        max_time = parse_time(
            policy.get("maxTime")
        )

        contamination_threshold = policy.get(
            "contaminationThreshold"
        )

        if (
            min_time is not None
            and max_time is not None
            and min_time <= max_time
            and isinstance(
                contamination_threshold,
                (int, float)
            )
            and not isinstance(
                contamination_threshold,
                bool
            )
            and isfinite(
                contamination_threshold
            )
            and 0 <= contamination_threshold <= 1
        ):
            policy_valid = True


    candidates = []

    for row in retained_rows:

        if not policy_valid:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "POLICY_INVALID"
                ]
            })

            continue


        row_time = parse_time(
            row["eventTime"]
        )

        if (
            row_time < min_time
            or row_time > max_time
        ):

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": [
                    "OUT_OF_WINDOW"
                ]
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

        digest = hashlib.sha256(
            row["entity"].encode("utf-8")
        ).digest()

        bucket = digest[0] % 10

        if bucket <= 5:
            splits["train"].append(row)

        elif bucket <= 7:
            splits["validation"].append(row)

        else:
            splits["test"].append(row)


    # ========================================================
    # TRAIN CONTAMINATION
    # ========================================================

    train_word_sets = [
        word_set(row["text"])
        for row in splits["train"]
    ]


    for split_name in (
        "validation",
        "test"
    ):

        kept = []

        for row in splits[split_name]:

            current_words = word_set(
                row["text"]
            )

            contaminated = False

            for train_words in train_word_sets:

                similarity = jaccard(
                    current_words,
                    train_words
                )

                if (
                    similarity
                    >= contamination_threshold
                ):
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

        splits[split_name].sort(
            key=lambda row: (
                utf8(row["id"]),
                compact_json(row).encode("utf-8")
            )
        )

        lines = []

        for row in splits[split_name]:

            lines.append(
                compact_json(row)
                + "\n"
            )

        artifact = "".join(
            lines
        ).encode("utf-8")

        digests[split_name] = hashlib.sha256(
            artifact
        ).hexdigest()


    # ========================================================
    # SORT REJECTED OBJECTS
    # ========================================================

    rejected_objects.sort(
        key=lambda item: (
            (
                utf8(item["uri"])
                if isinstance(item["uri"], str)
                else b""
            ),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # SORT REJECTED ROWS
    # ========================================================

    rejected_rows.sort(
        key=lambda item: (
            utf8(item["id"]),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # SORT LINEAGE
    # ========================================================

    lineage.sort(
        key=lambda item: (
            utf8(item["uri"]),
            compact_json(item).encode("utf-8")
        )
    )


    # ========================================================
    # RESPONSE
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
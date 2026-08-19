from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import hashlib
import unicodedata
import re
from datetime import datetime, timezone, timedelta
from math import isfinite

app = FastAPI()

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

# ---------- CRC32C ----------

POLY = 0x82F63B78
CRC_TABLE = []

for i in range(256):
    c = i
    for _ in range(8):
        c = (c >> 1) ^ POLY if c & 1 else c >> 1
    CRC_TABLE.append(c)


def crc32c(data: bytes):
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


# ---------- Helpers ----------

def utf8_key(s):
    return s.encode("utf-8")


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def canonical_text(s):
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", s)


TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"
)


def parse_time(s):
    if not isinstance(s, str) or not TIME_RE.match(s):
        return None

    try:
        if s.endswith("Z"):
            iso = s[:-1] + "+00:00"
        else:
            iso = s

        dt = datetime.fromisoformat(iso)

        offset = dt.utcoffset()
        if offset is None:
            return None

        total = abs(int(offset.total_seconds() // 60))
        hours = total // 60
        minutes = total % 60

        if hours > 14 or (hours == 14 and minutes != 0):
            return None

        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def utc_string(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def word_set(text):
    return {
        "".join(chars)
        for chars in re.findall(r"\w+", text, flags=re.UNICODE)
        if any(ch.isalnum() for ch in chars)
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------- Endpoint ----------

@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    policy = body.get("policy")
    objects = body.get("objects")

    if policy is None or not isinstance(objects, list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    rejected_objects = []
    rejected_rows = []
    lineage = []
    parsed_rows = []

    # ---------- Objects ----------

    for obj in objects:
        if not isinstance(obj, dict):
            rejected_objects.append({"uri": None, "reasonCodes": ["URI_INVALID"]})
            continue

        uri = obj.get("uri")
        reasons = []

        if not isinstance(uri, str) or not re.fullmatch(r"gs://[^/]+/.+", uri):
            reasons.append("URI_INVALID")

        generation = obj.get("generation")
        fetched = obj.get("fetchedGeneration")

        gen_valid = isinstance(generation, str) and generation.isdecimal()
        fetched_valid = isinstance(fetched, str) and fetched.isdecimal()

        if not gen_valid or not fetched_valid:
            reasons.append("GENERATION_INVALID")

        if generation != fetched:
            reasons.append("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        crc_valid = isinstance(crc, str) and re.fullmatch(r"[0-9a-f]{8}", crc)

        if not crc_valid:
            reasons.append("CRC32C_INVALID")

        content = obj.get("content")

        if isinstance(content, str) and crc_valid:
            actual = f"{crc32c(content.encode('utf-8')):08x}"
            if actual != crc:
                reasons.append("CRC32C_MISMATCH")

        schema = obj.get("schemaId")

        if not isinstance(content, str) or schema != "training-v1":
            reasons.append("SCHEMA_INVALID")

        rows = []

        if isinstance(content, str):
            try:
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
            except Exception:
                reasons.append("JSONL_INVALID")
                rows = []

            if not rows:
                reasons.append("SCHEMA_INVALID")

        valid_rows = []

        if not reasons:
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}
                    or not isinstance(row["id"], str)
                    or not isinstance(row["entity"], str)
                    or not isinstance(row["eventTime"], str)
                    or not isinstance(row["text"], str)
                    or not isinstance(row["revision"], int)
                    or isinstance(row["revision"], bool)
                    or row["revision"] < 0
                    or row["revision"] > 9007199254740991
                    or parse_time(row["eventTime"]) is None
                ):
                    reasons.append("SCHEMA_INVALID")
                    break

                valid_rows.append(row)

        if reasons:
            rejected_objects.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sorted(set(reasons), key=utf8_key)
            })
            continue

        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": crc,
            "schemaId": schema
        })

        for row in valid_rows:
            parsed_rows.append({
                "id": row["id"],
                "entity": canonical_text(row["entity"]),
                "eventTime": utc_string(parse_time(row["eventTime"])),
                "revision": row["revision"],
                "text": canonical_text(row["text"])
            })

    # ---------- Dedup ----------

    groups = {}

    for row in parsed_rows:
        key = compact([row["entity"], row["eventTime"], row["text"]])
        groups.setdefault(key, []).append(row)

    retained = []

    for rows in groups.values():
        rows.sort(key=lambda r: (-r["revision"], utf8_key(r["id"])))
        retained.append(rows[0])

        for loser in rows[1:]:
            rejected_rows.append({
                "id": loser["id"],
                "reasonCodes": ["DUPLICATE"]
            })

    # ---------- Policy ----------

    policy_valid = False
    min_dt = max_dt = None
    threshold = None

    if isinstance(policy, dict):
        min_dt = parse_time(policy.get("minTime"))
        max_dt = parse_time(policy.get("maxTime"))
        threshold = policy.get("contaminationThreshold")

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
            rejected_rows.append({"id": row["id"], "reasonCodes": ["POLICY_INVALID"]})
        else:
            dt = parse_time(row["eventTime"])
            if dt < min_dt or dt > max_dt:
                rejected_rows.append({"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
            else:
                candidates.append(row)

    # ---------- Split ----------

    splits = {"train": [], "validation": [], "test": []}

    for row in candidates:
        b = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0] % 10

        if b <= 5:
            splits["train"].append(row)
        elif b <= 7:
            splits["validation"].append(row)
        else:
            splits["test"].append(row)

    train_words = [word_set(r["text"]) for r in splits["train"]]

    for split_name in ("validation", "test"):
        kept = []

        for row in splits[split_name]:
            ws = word_set(row["text"])

            contaminated = any(
                jaccard(ws, tw) >= threshold
                for tw in train_words
            )

            if contaminated:
                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": ["TRAIN_CONTAMINATION"]
                })
            else:
                kept.append(row)

        splits[split_name] = kept

    # ---------- Deterministic artifacts ----------

    digests = {}

    for name in splits:
        splits[name].sort(
            key=lambda r: (utf8_key(r["id"]), compact(r).encode("utf-8"))
        )

        payload = "".join(compact(r) + "\n" for r in splits[name]).encode("utf-8")
        digests[name] = hashlib.sha256(payload).hexdigest()

    rejected_objects.sort(
        key=lambda x: (
            utf8_key(x["uri"]) if isinstance(x["uri"], str) else b"",
            compact(x).encode("utf-8")
        )
    )

    rejected_rows.sort(
        key=lambda x: (utf8_key(x["id"]), compact(x).encode("utf-8"))
    )

    lineage.sort(
        key=lambda x: (utf8_key(x["uri"]), compact(x).encode("utf-8"))
    )

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage
    }
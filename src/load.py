

import gzip
import json
import logging
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

logger = logging.getLogger(__name__)

REQUIRED_TOP_LEVEL_KEYS = {
    "candidate_id", "profile", "career_history", "education",
    "skills", "redrob_signals",
}

REQUIRED_PROFILE_KEYS = {
    "anonymized_name", "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_company_size", "current_industry",
}

def _validate_record(record: dict, line_num: int) -> Optional[str]:
    
    missing_top = REQUIRED_TOP_LEVEL_KEYS - set(record.keys())
    if missing_top:
        return f"Line {line_num}: missing top-level keys: {missing_top}"

    profile = record.get("profile", {})
    missing_profile = REQUIRED_PROFILE_KEYS - set(profile.keys())
    if missing_profile:
        return f"Line {line_num}: missing profile keys: {missing_profile}"

    cid = record.get("candidate_id", "")
    if not cid or not cid.startswith("CAND_"):
        return f"Line {line_num}: invalid candidate_id: {cid!r}"

    return None

def stream_jsonl(filepath: str) -> Generator[Dict[str, Any], None, None]:
    
    path = Path(filepath)

    if path.suffix == ".json":
        # JSON array — must load entirely, but then yield one-by-one
        logger.info(f"Loading JSON array from {path.name} ...")
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        if not isinstance(candidates, list):
            raise ValueError(f"Expected JSON array, got {type(candidates).__name__}")
        logger.info(f"Loaded {len(candidates)} candidates from JSON array")
        for i, record in enumerate(candidates, 1):
            err = _validate_record(record, i)
            if err:
                logger.warning(f"Skipping invalid record: {err}")
                continue
            yield record
        return

    # JSONL or JSONL.GZ
    opener = gzip.open if path.name.endswith(".gz") else open
    mode = "rt" if path.name.endswith(".gz") else "r"

    logger.info(f"Streaming JSONL from {path.name} ...")
    valid_count = 0
    error_count = 0

    with opener(path, mode, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                if HAS_ORJSON:
                    record = orjson.loads(line)
                else:
                    record = json.loads(line)
            except Exception as e:
                logger.warning(f"Line {line_num}: JSON parse error: {e}")
                error_count += 1
                continue

            err = _validate_record(record, line_num)
            if err:
                logger.warning(f"Skipping invalid record: {err}")
                error_count += 1
                continue

            valid_count += 1
            yield record

    logger.info(
        f"Streamed {valid_count} valid candidates "
        f"({error_count} errors skipped) from {path.name}"
    )

def load_all(filepath: str) -> List[Dict[str, Any]]:
    
    return list(stream_jsonl(filepath))

def count_candidates(filepath: str) -> int:
    
    path = Path(filepath)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f))

    opener = gzip.open if path.name.endswith(".gz") else open
    mode = "rt" if path.name.endswith(".gz") else "r"
    count = 0
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

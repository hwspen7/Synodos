"""Conservative time-aligned ASR deduplication."""
from __future__ import annotations
import unicodedata
from difflib import SequenceMatcher

def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())

def is_temporal_duplicate(start: int, end: int, text: str, confirmed: list[dict]) -> bool:
    if start < 0 or end <= start:
        return False
    left = normalized_text(text)
    if not left:
        return False
    for item in confirmed[-64:]:
        a = int(item.get("start_sample", -1))
        b = int(item.get("end_sample", -1))
        duration = min(end - start, b - a)
        overlap = max(0, min(end, b) - max(start, a))
        if duration < 1 or overlap / duration < 0.85:
            continue
        right = normalized_text(str(item.get("text", "")))
        if right and SequenceMatcher(None, left[:2000], right[:2000], autojunk=False).ratio() >= 0.92:
            return True
    return False

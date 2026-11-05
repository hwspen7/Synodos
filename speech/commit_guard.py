"""Conservative time-aligned ASR deduplication."""
from __future__ import annotations
import re
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

def is_adjacent_tail_echo(
    start: int, end: int, text: str, confirmed: list[dict],
    *, sample_rate: int = 16000, timestamp_quality: str = "segment",
) -> bool:
    """Flag a tiny unaligned repeat of the *previous sentence's last word*.

    A new recording of a valid short reply must not be deleted simply
    because its words repeat a past utterance. This narrow rule only
    applies to unaligned single-word echoes at the same sound boundary.
    It does not attempt unsafe translation/transliteration matching.
    """
    if (sample_rate <= 0 or start < 0 or end <= start
            or end - start > round(1.05 * sample_rate)):
        return False
    # Only one short Latin word; full sentences and CJK code switching
    # cannot be judged duplicative by a text suffix alone.
    fragment = re.findall(r"[a-z0-9]+", unicodedata.normalize(
        "NFKC", str(text or "")
    ).casefold())
    if len(fragment) != 1 or len(fragment[0]) > 14:
        return False
    for previous in reversed(confirmed[-8:]):
        old_start = int(previous.get("start_sample", -1))
        old_end = int(previous.get("end_sample", -1))
        if old_start < 0 or old_end - old_start < round(0.7 * sample_rate):
            continue
        if timestamp_quality == "word":
            # Precise, independently positioned words are preserved. Only
            # the same previous word's *acoustic interval* can be an echo.
            prior_words = previous.get("words") or []
            if not prior_words:
                continue
            last = prior_words[-1]
            if not isinstance(last, dict):
                continue
            last_start = int(last.get("start_ms", -100000)) * sample_rate // 1000
            last_end = int(last.get("end_ms", -100000)) * sample_rate // 1000
            base = int(previous.get("_audio_base_sample", 0))
            last_start += base
            last_end += base
            if (abs(start - last_start) > round(0.18 * sample_rate)
                    or abs(end - last_end) > round(0.23 * sample_rate)):
                continue
        elif (start < old_end - round(0.18 * sample_rate)
              or start > old_end + round(0.24 * sample_rate)
              or end > old_end + round(0.96 * sample_rate)):
            continue
        old_text = str(previous.get("text") or "").strip()
        if not old_text.endswith((".", "?", "!", "\u3002", "\uff1f", "\uff01")):
            continue
        tokens = re.findall(
            r"[a-z0-9]+", unicodedata.normalize("NFKC", old_text).casefold()
        )
        if len(tokens) >= 3 and tokens[-1] == fragment[0]:
            return True
    return False

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

def _words_only(text: str) -> str:
    return " ".join("".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in normalized_text(text)
    ).split())


def is_adjacent_tail_echo(
    start: int, end: int, text: str, confirmed: list[dict],
    *, sample_rate: int = 16000, timestamp_quality: str = "segment",
) -> bool:
    """Suppress a repeated suffix only when both records own the same audio."""
    if sample_rate <= 0 or start < 0 or not 0 < end - start <= 1.05 * sample_rate:
        return False
    fragment = _words_only(text)
    if not fragment or len(fragment) > 32:
        return False
    for previous in reversed(confirmed[-8:]):
        old_end = int(previous.get("end_sample", -1))
        if start >= old_end:
            continue
        old_start = int(previous.get("start_sample", -1))
        base = int(previous.get("_audio_base_sample", 0))
        # Without a prior word alignment, proximity to an utterance end
        # cannot establish ownership of a new person's short response.
        words = previous.get("words") or ()
        if not words:
            continue
        suffix = []
        try:
            last = base + int(words[-1]["end_ms"]) * sample_rate // 1000
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        for word in reversed(words[-8:]):
            try:
                left = base + int(word["start_ms"]) * sample_rate // 1000
                right = base + int(word["end_ms"]) * sample_rate // 1000
                suffix.insert(0, str(word["text"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                break
            if not old_start <= left < right <= last <= old_end:
                break
            overlap = min(end, last) - max(start, left)
            if (overlap >= 0.70 * max(end - start, last - left)
                    and fragment in (_words_only("".join(suffix)),
                                     _words_only(" ".join(suffix)))):
                return True
    return False

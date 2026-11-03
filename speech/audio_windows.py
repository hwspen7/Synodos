"""Bounded speech windows with absolute sample offsets for multilingual ASR.

The inputs are acoustic VAD intervals, not text or inferred languages. Each
window may be classified independently by an ASR backend. Avoid making a
meeting-wide language decision from its opening seconds.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def speech_windows(
    regions: Iterable[Mapping[str, int]],
    audio_samples: int,
    *,
    sample_rate: int = 16000,
    merge_gap_ms: int = 320,
    pad_ms: int = 230,
    max_window_s: int = 18,
) -> list[tuple[int, int]]:
    if sample_rate <= 0 or audio_samples < 0:
        raise ValueError("Invalid audio geometry")
    if not 0 <= merge_gap_ms <= 3000 or not 0 <= pad_ms <= 2000:
        raise ValueError("Invalid window parameters")
    if not 3 <= max_window_s <= 120:
        raise ValueError("Invalid maximum window size")
    spans = []
    for item in regions:
        try:
            a, b = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        a, b = max(0, min(audio_samples, a)), max(0, min(audio_samples, b))
        if a < b:
            spans.append((a, b))
    spans.sort()
    gap_samples = merge_gap_ms * sample_rate // 1000
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= gap_samples:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    pad = pad_ms * sample_rate // 1000
    max_samples = max_window_s * sample_rate
    result: list[tuple[int, int]] = []
    for a, b in merged:
        # Keep a small acoustic margin without making adjacent windows overlap.
        start, end = max(0, a - pad), min(audio_samples, b + pad)
        if result and start < result[-1][1]:
            start = result[-1][1]
        while end - start > max_samples:
            result.append((start, start + max_samples))
            start += max_samples
        if end > start:
            result.append((start, end))
    return result


def local_speech_regions(regions: Iterable[Mapping[str, int]], start: int, end: int) -> list[dict[str, int]]:
    """Clip global regions into a window, translating offsets to zero."""
    if end <= start:
        return []
    output = []
    for region in regions:
        a, b = max(start, int(region["start"])), min(end, int(region["end"]))
        if b > a:
            output.append({"start": a - start, "end": b - start})
    return output

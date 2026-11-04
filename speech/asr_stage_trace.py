"""Transcript-free, frame-independent ASR stage interval audit.

Intervals reveal where potential spoken audio stopped being represented;
they cannot prove word correctness or infer an ASR transcript.
"""
from __future__ import annotations

import math


def merged_intervals(intervals, *, duration_ms: int) -> list[list[int]]:
    if type(duration_ms) is not int or duration_ms < 0:
        raise ValueError("Invalid audio interval duration")
    candidates = []
    for item in intervals:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        a, b = item
        if (not isinstance(a, (int, float)) or
                not isinstance(b, (int, float)) or
                not math.isfinite(a) or not math.isfinite(b)):
            continue
        start = max(0, min(duration_ms, int(a)))
        end = max(0, min(duration_ms, int(b)))
        if end > start:
            candidates.append((start, end))
    candidates.sort()
    merged = []
    for a, b in candidates:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def model_intervals(pieces, *, duration_ms: int,
                    shift_ms: int = 0) -> list[list[int]]:
    """Read raw model timestamps; never store text in diagnostic traces."""
    rows = []
    for piece in pieces:
        start, end = getattr(piece, "start", None), getattr(piece, "end", None)
        if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and math.isfinite(start) and math.isfinite(end)):
            rows.append((
                round(float(start) * 1000) + shift_ms,
                round(float(end) * 1000) + shift_ms,
            ))
    return merged_intervals(rows, duration_ms=duration_ms)


def utterance_intervals(items, *, duration_ms: int) -> list[list[int]]:
    return merged_intervals(
        ((getattr(item, "start_ms", None),
          getattr(item, "end_ms", None)) for item in items),
        duration_ms=duration_ms,
    )


def shifted(intervals, offset_ms: int) -> list[list[int]]:
    return [[a + offset_ms, b + offset_ms] for a, b in intervals]

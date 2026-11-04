"""Exact acoustically supported time intervals for recognized caption speakers.

The final Nemotron turn list is the ONLY authority for speaker activity.
A caption is a *text* hypothesis and may include nearby silence; never
pretend its entire ASR sentence time range is one continuous voice turn.
This module only annotates existing caption text; it neither truncates
ASR text nor invents timestamps or speaker identities.
"""
from __future__ import annotations


def speaker_support_intervals(
    turns: list[dict], start_ms: int, end_ms: int, speaker_id: int,
    *, merge_gap_ms: int = 0, max_intervals: int = 32,
) -> list[list[int]]:
    """Intersect one finalized speaker channel with one caption interval.

    Multiple disjoint spans are retained, including pauses and overlaps.
    Missing/noisy label data fails closed. No RTTM or human reference is used.
    """
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or end_ms <= start_ms
            or type(speaker_id) is not int or speaker_id <= 0
            or not 0 <= merge_gap_ms <= 250
            or not 1 <= max_intervals <= 256):
        raise ValueError("Invalid caption support geometry")
    if not isinstance(turns, list):
        raise ValueError("Expected finalized speaker turns")
    parts = []
    for row in turns:
        if not isinstance(row, dict):
            raise ValueError("Malformed finalized speaker turn")
        channel, start, end = (
            row.get("speaker_id"), row.get("start_ms"), row.get("end_ms"),
        )
        if (type(channel) is not int or not 1 <= channel <= 8
                or type(start) is not int or type(end) is not int
                or start < 0 or end <= start):
            raise ValueError("Invalid finalized speaker turn")
        if channel != speaker_id:
            continue
        left = max(start_ms, start)
        right = min(end_ms, end)
        if right > left:
            parts.append((left, right))
    parts.sort()
    merged = []
    for a, b in parts:
        if merged and a - merged[-1][1] <= merge_gap_ms:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    # Return only complete evidence; silently truncating would turn into
    # false missed speaker-time in downstream consumers.
    if len(merged) > max_intervals:
        raise ValueError("Too many disjoint speaker activity intervals")
    return merged


def caption_support_summary(
    turns: list[dict], start_ms: int, end_ms: int, speaker_id: int,
) -> dict:
    """Non-destructive caption activity metadata for the live UI and audit."""
    windows = speaker_support_intervals(
        turns, start_ms, end_ms, speaker_id,
    )
    fraction = round(
        sum(b - a for a, b in windows) / (end_ms - start_ms), 5,
    )
    return {
        "speaker_activity_intervals_ms": windows,
        "speaker_activity_fraction": fraction,
        "speaker_activity_time_source": "nemotron3_official_turns",
    }

"""Conservative promotion of new anonymous diarization clusters.

A new positive Speaker ID is a claim of repeatable acoustic identity.
A singleton short or coarsely timed ASR segment is only an unconfirmed
candidate. Its recognized text remains intact and separately saved.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def tentative_new_clusters(
    segments: Iterable[Mapping], approved_ids: set[int],
    sample_rate: int = 16000,
) -> set[int]:
    """Raw cluster labels still lacking independent identification evidence."""
    if sample_rate <= 0:
        raise ValueError("Invalid acoustic sample rate")
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for seg in segments:
        raw = seg.get("cluster")
        if type(raw) is int and raw >= 0:
            grouped[raw].append(seg)
    tentative: set[int] = set()
    for raw, members in grouped.items():
        if any(type(s.get("speaker_id")) is int and
               s["speaker_id"] in approved_ids for s in members):
            continue
        reliable = [
            s for s in members
            if s.get("emb") is not None and
            not s.get("speaker_uncertain", False) and
            int(s.get("end_sample") or 0) > int(s.get("start_sample") or 0)
        ]
        # At least two non-overlapping acoustic observations must agree.
        # Don't count multiple overlapping predictions of one utterance as
        # corroboration merely because they have different segment IDs.
        spans = sorted(
            (int(s["start_sample"]), int(s["end_sample"]))
            for s in reliable
        )
        independent = 0
        prior_end = -1
        for start, end in spans:
            if start >= prior_end:
                independent += 1
                prior_end = end
        corroborated = independent >= 2
        # A long, precisely word-aligned interval offers stronger evidence
        # than a Qwen VAD-wide approximate timestamp.
        long_verified = any(
            int(s["end_sample"]) - int(s["start_sample"])
            >= round(3.5 * sample_rate)
            and s.get("timestamp_quality") == "word"
            and len(s.get("words") or ()) >= 3
            for s in reliable
        )
        if not corroborated and not long_verified:
            tentative.add(raw)
    return tentative

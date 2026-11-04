"""Reference-free risk inventory for overlapping voices in confirmed captions.

Acoustic activity cannot prove that intelligible words were omitted. We
measure how often two or more final Nemotron speaker channels were active
without corresponding parallel primary-speaker caption coverage. This is
review evidence, NOT an estimate of word accuracy or diarization DER.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def overlap_content_risk(
    acoustic_turns: Sequence[Mapping],
    caption_turns: Sequence[Mapping],
    *, acoustic_origin_ms: int = 0, speaker_id_offset: int = 0,
    min_overlap_ms: int = 120, max_review_spans: int = 32,
) -> dict:
    if (type(acoustic_origin_ms) is not int or acoustic_origin_ms < 0
            or type(speaker_id_offset) is not int
            or not 20 <= min_overlap_ms <= 2000
            or not 1 <= max_review_spans <= 256):
        raise ValueError("Invalid overlap audit configuration")
    if not isinstance(acoustic_turns, (list, tuple)):
        raise ValueError("Acoustic turns must be finalized")
    if not isinstance(caption_turns, (list, tuple)):
        raise ValueError("Caption evidence must be a list")
    intervals = []
    for turn in acoustic_turns:
        sid, a, b = (
            turn.get("speaker_id"), turn.get("start_ms"), turn.get("end_ms")
        )
        if (type(sid) is not int or not 1 <= sid <= 8
                or type(a) is not int or type(b) is not int
                or a < 0 or b <= a):
            raise ValueError("Invalid final acoustic turn")
        intervals.append((a + acoustic_origin_ms, b + acoustic_origin_ms,
                          sid + speaker_id_offset))
    captions = []
    for turn in caption_turns:
        a, b, sid = (
            turn.get("start_ms"), turn.get("end_ms"),
            turn.get("speaker_id"),
        )
        if (type(a) is not int or type(b) is not int
                or a < 0 or b <= a or type(sid) is not int):
            raise ValueError("Invalid final caption")
        if sid > 0:
            captions.append((a, b, sid))
    if not intervals:
        return {
            "overlap_acoustic_clock_ms": 0,
            "overlap_clock_no_caption_ms": 0,
            "overlap_clock_underrepresented_ms": 0,
            "review_spans": [], "note": "No finalized overlapping voice evidence",
        }

    boundaries = sorted({
        clock for start, end, _ in intervals + captions
        for clock in (start, end)
    })
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        acoustic = {sid for a, b, sid in intervals if a < end and b > start}
        if len(acoustic) < 2:
            continue
        published = {sid for a, b, sid in captions if a < end and b > start}
        state = ("no_caption" if not published
                 else "underrepresented" if not acoustic.issubset(published)
                 else "represented")
        if (spans and spans[-1]["end_ms"] == start
                and spans[-1]["status"] == state
                and spans[-1]["speaker_ids"] == sorted(acoustic)):
            spans[-1]["end_ms"] = end
        else:
            spans.append({
                "start_ms": start, "end_ms": end,
                "speaker_ids": sorted(acoustic),
                "status": state,
            })
    verified_spans = [
        {**row, "duration_ms": row["end_ms"] - row["start_ms"]}
        for row in spans if row["end_ms"] - row["start_ms"] >= min_overlap_ms
    ]
    return {
        "overlap_acoustic_clock_ms": sum(
            s["duration_ms"] for s in verified_spans
        ),
        "overlap_clock_no_caption_ms": sum(
            s["duration_ms"] for s in verified_spans
            if s["status"] == "no_caption"
        ),
        "overlap_clock_underrepresented_ms": sum(
            s["duration_ms"] for s in verified_spans
            if s["status"] == "underrepresented"
        ),
        "review_spans": sorted(
            verified_spans,
            key=lambda row: (-row["duration_ms"], row["start_ms"]),
        )[:max_review_spans],
        "note": (
            "Reference-free model evidence. Unrepresented speaker-time "
            "does not prove intelligible words were omitted; no text was "
            "reconstructed or assigned to a second voice."
        ),
    }

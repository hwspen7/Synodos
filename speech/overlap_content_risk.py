"""Reference-free risk inventory for overlapping voices in confirmed captions.

Acoustic activity cannot prove that intelligible words were omitted. We
measure how often two or more final Nemotron speaker channels were active
without corresponding parallel primary-speaker caption coverage. This is
review evidence, NOT an estimate of word accuracy or diarization DER.
"""
from __future__ import annotations

from collections import defaultdict
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

    # O(N log N) sweep rather than rescanning all turns per boundary.
    # Counters preserve a speaker even when overlapping adjacent turns for
    # that same channel have independent start/end events.
    changes = defaultdict(lambda: [defaultdict(int), defaultdict(int)])
    for a, b, sid in intervals:
        changes[a][0][sid] += 1
        changes[b][0][sid] -= 1
    for a, b, sid in captions:
        changes[a][1][sid] += 1
        changes[b][1][sid] -= 1
    timeline = sorted(changes)
    active_acoustic = defaultdict(int)
    active_caption = defaultdict(int)
    spans = []
    for start, end in zip(timeline, timeline[1:]):
        for speaker, delta in changes[start][0].items():
            active_acoustic[speaker] += delta
        for speaker, delta in changes[start][1].items():
            active_caption[speaker] += delta
        acoustic = {sid for sid, n in active_acoustic.items() if n > 0}
        if len(acoustic) < 2:
            continue
        published = {sid for sid, n in active_caption.items() if n > 0}
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

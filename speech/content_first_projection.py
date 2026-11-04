"""Non-destructive, content-first projection of confirmed ASR into diarized cues.

The original ASR segment remains the source of truth for editing, translation,
persistence, and exact text. Word timings are optional acoustic *evidence*,
not permission to truncate, repeat, or fabricate any recognized word.

Official Nemotron turns may overlap; a recognized word is assigned to at most
one primary speaker. Additional simultaneously active speakers are reported
as overlap evidence but NEVER given the same word as a second transcript.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def _plain(text: str) -> str:
    # Literal identity matters in multilingual transcripts. NFKC would
    # equate distinct glyphs (e.g. Roman numerals) and is too permissive.
    return text.strip()


def _valid_clock(value: object) -> bool:
    return type(value) is int and value >= 0


def _union_duration(spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def _speaker_cover(turns: Sequence[Mapping], a: int, b: int) -> dict[int, int]:
    cover: dict[int, list[tuple[int, int]]] = {}
    for turn in turns:
        speaker, start, end = (turn.get("speaker_id"),
                               turn.get("start_ms"), turn.get("end_ms"))
        if (type(speaker) is not int or not 1 <= speaker <= 8
                or not _valid_clock(start) or not _valid_clock(end)
                or end <= start):
            raise ValueError("Invalid official speaker activity")
        left, right = max(a, start), min(b, end)
        if left < right:
            cover.setdefault(speaker, []).append((left, right))
    return {s: _union_duration(spans) for s, spans in cover.items()}


def _overlap_intervals(
    turns: Sequence[Mapping], a: int, b: int, *,
    min_ms: int = 120, limit: int = 24,
) -> list[dict]:
    points = {a, b}
    for turn in turns:
        start, end = turn["start_ms"], turn["end_ms"]
        if end > a and start < b:
            points.add(max(a, start))
            points.add(min(b, end))
    ordered = sorted(points)
    result: list[dict] = []
    for start, end in zip(ordered, ordered[1:]):
        speakers = sorted(_speaker_cover(turns, start, end))
        if len(speakers) < 2:
            continue
        if (result and result[-1]["end_ms"] == start
                and result[-1]["speaker_ids"] == speakers):
            result[-1]["end_ms"] = end
        else:
            result.append({
                "start_ms": start, "end_ms": end, "speaker_ids": speakers,
            })
    return [
        item for item in result if item["end_ms"] - item["start_ms"] >= min_ms
    ][:limit]


def project_confirmed_caption(
    *, text: str, words: object, caption_start_ms: int, caption_end_ms: int,
    word_base_ms: int | None, official_turns: Sequence[Mapping],
    acoustic_origin_ms: int = 0, speaker_id_offset: int = 0,
    min_word_fraction: float = 0.55, min_margin: float = 0.20,
    max_words: int = 256, max_cues: int = 32,
) -> dict:
    """Create optional speaker-attributed display cues with exact text conservation.

    Word times are relative to the inference audio window's `word_base_ms`.
    Official speaker times are relative to `acoustic_origin_ms` (a separate
    continuously received audio stream). Both are projected into *absolute*
    meeting milliseconds. Old resumed segments without a trusted word base
    cannot be split and are displayed canonically.

    All failures return the original caption unchanged, with a reason.
    """
    if (not isinstance(text, str)
            or not _valid_clock(caption_start_ms)
            or not _valid_clock(caption_end_ms)
            or caption_end_ms <= caption_start_ms
            or not _valid_clock(acoustic_origin_ms)
            or type(speaker_id_offset) is not int
            or not 0 <= min_word_fraction <= 1
            or not 0 <= min_margin <= 1):
        raise ValueError("Invalid caption projection arguments")
    if not isinstance(official_turns, (list, tuple)):
        raise ValueError("Official speaker turns are required")
    # Scan the whole finalized timeline only once per caption. Individual
    # word decisions and overlap boundaries use nearby turns exclusively.
    # This prevents quadratic work on hours-long meetings.
    acoustic_start = max(0, caption_start_ms - acoustic_origin_ms)
    acoustic_end = max(acoustic_start, caption_end_ms - acoustic_origin_ms)
    nearby_turns = []
    for turn in official_turns:
        if not isinstance(turn, Mapping):
            raise ValueError("Malformed official speaker turn")
        sid, a, b = (
            turn.get("speaker_id"), turn.get("start_ms"), turn.get("end_ms"),
        )
        if (type(sid) is not int or not 1 <= sid <= 8
                or not _valid_clock(a) or not _valid_clock(b) or b <= a):
            raise ValueError("Invalid finalized Nemotron speaker turn")
        if b > acoustic_start and a < acoustic_end:
            nearby_turns.append(turn)
    overlap = _overlap_intervals(
        nearby_turns, acoustic_start, max(acoustic_start + 1, acoustic_end),
    )
    for interval in overlap:
        interval["start_ms"] += acoustic_origin_ms
        interval["end_ms"] += acoustic_origin_ms
        interval["speaker_ids"] = [
            sid + speaker_id_offset for sid in interval["speaker_ids"]
        ]

    baseline = {
        "cues": [], "overlap_intervals_ms": overlap,
        "status": "unaligned", "text_preserved": True,
    }
    if (not text.strip() or word_base_ms is None or
            not _valid_clock(word_base_ms)
            or not isinstance(words, (list, tuple))
            or not 1 <= len(words) <= max_words):
        return baseline
    parsed = []
    for item in words:
        if not isinstance(item, Mapping):
            return baseline
        raw, a, b = item.get("text"), item.get("start_ms"), item.get("end_ms")
        if (not isinstance(raw, str) or not raw.strip()
                or not _valid_clock(a) or not _valid_clock(b) or b <= a):
            return baseline
        abs_a, abs_b = a + word_base_ms, b + word_base_ms
        if abs_a < caption_start_ms - 150 or abs_b > caption_end_ms + 150:
            return baseline
        if parsed and abs_a < parsed[-1][1]:
            return baseline
        parsed.append((raw, abs_a, abs_b))
    # Require exact reconstruction after only stripping external whitespace:
    # no punctuation, word or whitespace internal to the transcript is lost.
    if _plain("".join(word[0] for word in parsed)) != _plain(text):
        return baseline
    if len(parsed) < 2:
        return {**baseline, "status": "single_word"}

    cues = []
    current = None
    for raw, a, b in parsed:
        scores = sorted(
            _speaker_cover(
                nearby_turns,
                max(0, a - acoustic_origin_ms),
                max(1, b - acoustic_origin_ms),
            ).items(), key=lambda pair: (-pair[1], pair[0]),
        )
        width = b - a
        top = scores[0][1] / width if scores else 0.0
        next_score = scores[1][1] / width if len(scores) > 1 else 0.0
        sid = (
            scores[0][0] + speaker_id_offset
            if top >= min_word_fraction and top - next_score >= min_margin
            else -1
        )
        if current is None or current["speaker_id"] != sid:
            current = {
                "speaker_id": sid, "speaker_name": (
                    f"Speaker {sid}" if sid > 0 else "Unidentified Speaker"
                ),
                "start_ms": a, "end_ms": b, "text": raw,
                "word_count": 1,
            }
            cues.append(current)
        else:
            current["end_ms"] = b
            current["text"] += raw
            current["word_count"] += 1
    if len(cues) > max_cues:
        return {**baseline, "status": "too_many_speaker_changes"}
    # The cue text sequence is exactly the source text, not a second ASR
    # hypothesis. This is mandatory before any display projection is emitted.
    if _plain("".join(item["text"] for item in cues)) != _plain(text):
        return {**baseline, "status": "content_mismatch"}
    return {
        "cues": cues if len(cues) > 1 else [],
        "overlap_intervals_ms": overlap,
        "status": "split_verified" if len(cues) > 1 else "one_speaker",
        "text_preserved": True,
    }

def project_caption_activity_for_audit(
    source_captions: Sequence[Mapping],
    display_projections: Mapping[int, Mapping],
) -> list[dict]:
    """Text-free actual displayed speaker activity, without duplicating parents.

    A verified split replaces the parent's one-speaker timing in the UI
    audit only. Unknown speaker cue spans remain unattributed. Canonical
    captions and the frozen DER scorer stay unchanged.
    """
    if not isinstance(source_captions, (list, tuple)):
        raise ValueError("Canonical captions must be ordered records")
    if not isinstance(display_projections, Mapping):
        raise ValueError("Projection index must be a mapping")
    rows: list[dict] = []
    for segment in source_captions:
        if not isinstance(segment, Mapping):
            raise ValueError("Malformed canonical caption")
        sid = segment.get("id")
        a, b, speaker = (
            segment.get("start_ms"), segment.get("end_ms"),
            segment.get("speaker_id"),
        )
        if (type(sid) is not int or type(a) is not int or type(b) is not int
                or a < 0 or b <= a or type(speaker) is not int):
            raise ValueError("Malformed canonical caption geometry")
        projection = display_projections.get(sid)
        cues = projection.get("cues") if isinstance(projection, Mapping) else None
        status = projection.get("status") if isinstance(projection, Mapping) else None
        if status == "split_verified" and isinstance(cues, list) and len(cues) > 1:
            for cue in cues:
                p, q, voice = (cue.get("start_ms"), cue.get("end_ms"),
                               cue.get("speaker_id"))
                if (type(p) is not int or type(q) is not int or
                        type(voice) is not int or p < a - 150 or q > b + 150
                        or q <= p):
                    raise ValueError("Malformed displayed speaker cue")
                rows.append({
                    "start_ms": p, "end_ms": q, "speaker_id": voice,
                })
        else:
            rows.append({
                "start_ms": a, "end_ms": b, "speaker_id": speaker,
            })
    return rows

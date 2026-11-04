"""Offline AMI DER and unidentified-segment audit for one immutable replay.

Scores the *published CAM++ live timeline* against pinned, human-derived
AMI RTTM reference turns. No model loads, speaker names, text or voiceprints
are written into the report. Requires diarization diagnostics on the server.
This is an approximate 100 ms, no-collar, overlap-aware DER, NOT a
model-independent speaker verification accuracy number.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.ami_shadow10_prepare import (
    DURATION_S, SELECTED, checksum, parse_reference, validate_reference,
)
from tools.phase07_benchmark import load_wav
from tools.speaker_stage1_compare import (
    _reference_turns, _score_diarization, baseline_turns,
)

SCHEMA = "meeting-ami-shadow10-der-audit-v1"


def reference_union_overlap(start: int, end: int, reference: list[dict]) -> int:
    """Return exact interval intersection with the union of all speech turns."""
    intervals = sorted(
        (max(start, x["start_ms"]), min(end, x["end_ms"]))
        for x in reference
        if x["start_ms"] < end and x["end_ms"] > start
    )
    total = 0
    merged_end = start
    for a, b in intervals:
        if a >= b:
            continue
        if a >= merged_end:
            total += b - a
            merged_end = b
        elif b > merged_end:
            total += b - merged_end
            merged_end = b
    return total


def unidentified_segments(timeline: dict, reference: list[dict],
                          duration_ms: int) -> dict:
    rows = timeline["segments"]
    unidentified = []
    for segment in rows:
        sid = segment.get("published_speaker_id")
        if type(sid) is int and sid > 0:
            continue
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if (type(start) is not int or type(end) is not int
                or start < 0 or end <= start or end > duration_ms + 30):
            raise ValueError("Invalid anonymous speaker interval")
        end = min(end, duration_ms)
        overlap = reference_union_overlap(start, end, reference)
        unidentified.append({
            "segment_id": segment.get("segment_id"),
            "start_ms": start,
            "end_ms": end,
            "reference_speech_overlap_ms": overlap,
            "reference_speech_overlap_fraction": round(
                overlap / (end - start), 4
            ),
            "embedding_available": bool(segment.get("embedding_available")),
            "speaker_uncertain": bool(segment.get("speaker_uncertain")),
            "speaker_evidence_status": segment.get(
                "speaker_evidence_status", "not_recorded"
            ),
        })
    return {
        "count": len(unidentified),
        "with_reference_speech": sum(
            row["reference_speech_overlap_ms"] > 0 for row in unidentified
        ),
        "outside_reference_speech": sum(
            row["reference_speech_overlap_ms"] == 0 for row in unidentified
        ),
        "intervals": unidentified,
        "interpretation": (
            "An unlabeled transcript interval overlapping annotated words "
            "requires audio review. RTTM only_words excludes some vocal "
            "events, and overlap does not prove an individual error cause."
        ),
    }


def missed_speaker_time_breakdown(
    timeline: dict, reference: list[dict], duration_ms: int,
    *, step_ms: int = 100,
) -> dict:
    """Partition missed reference speaker-time by visible transcript coverage.

    A missing label may coexist with an unidentified subtitle, with no
    visible subtitle, or with another known speaker during an overlap.
    These are descriptive time categories, not inferred root causes.
    """
    if step_ms != 100 or duration_ms <= 0:
        raise ValueError("This diagnostic uses fixed 100 ms scoring frames")
    intervals = timeline["segments"]
    counts = {
        "unidentified_transcript_ms": 0,
        "no_transcript_ms": 0,
        "simultaneous_speaker_shortfall_ms": 0,
        "total_missed_ms": 0,
        "reference_speaker_ms": 0,
    }
    for time_ms in range(0, duration_ms, step_ms):
        active_ref = {
            turn["speaker"] for turn in reference
            if turn["start_ms"] <= time_ms < turn["end_ms"]
        }
        if not active_ref:
            continue
        active_labeled = {
            row["published_speaker_id"] for row in intervals
            if type(row.get("published_speaker_id")) is int
            and row["published_speaker_id"] > 0
            and row["start_ms"] <= time_ms < row["end_ms"]
        }
        unidentified_visible = any(
            (type(row.get("published_speaker_id")) is not int
             or row["published_speaker_id"] <= 0)
            and row["start_ms"] <= time_ms < row["end_ms"]
            for row in intervals
        )
        missing = max(0, len(active_ref) - len(active_labeled))
        counts["reference_speaker_ms"] += len(active_ref) * step_ms
        counts["total_missed_ms"] += missing * step_ms
        if not missing:
            continue
        if not active_labeled:
            field = (
                "unidentified_transcript_ms"
                if unidentified_visible else "no_transcript_ms"
            )
        else:
            field = "simultaneous_speaker_shortfall_ms"
        counts[field] += missing * step_ms
    if (counts["unidentified_transcript_ms"]
            + counts["no_transcript_ms"]
            + counts["simultaneous_speaker_shortfall_ms"]
            != counts["total_missed_ms"]):
        raise AssertionError("Missed speaker-time categories must add up")
    return {
        "unidentified_transcript_s": round(
            counts["unidentified_transcript_ms"] / 1000, 2
        ),
        "without_transcript_s": round(
            counts["no_transcript_ms"] / 1000, 2
        ),
        "simultaneous_speaker_shortfall_s": round(
            counts["simultaneous_speaker_shortfall_ms"] / 1000, 2
        ),
        "total_missed_speaker_seconds": round(
            counts["total_missed_ms"] / 1000, 2
        ),
        "reference_speaker_seconds": round(
            counts["reference_speaker_ms"] / 1000, 2
        ),
        "note": (
            "All values are reference speaker-time, not distinct clock-time. "
            "No transcript at a frame may reflect ASR omission, boundaries, "
            "or processing. Simultaneous speaker shortfall does not "
            "establish a particular model failure cause."
        ),
    }


def reference_speech_without_captions(
    timeline: dict, reference: list[dict], duration_ms: int,
    *, frame_step_ms: int = 100,
) -> dict:
    """Locate reference speech not covered by ANY live transcript interval.

    This is ground-truth-only offline auditing, not a live VAD. Silence
    or imperfect annotation boundaries may make individual gaps debatable.
    """
    if frame_step_ms != 100 or duration_ms <= 0:
        raise ValueError("Only 100ms reference audit supported")
    frames = []
    for t in range(0, duration_ms, frame_step_ms):
        reference_active = any(
            row["start_ms"] <= t < row["end_ms"] for row in reference
        )
        caption_active = any(
            row["start_ms"] <= t < row["end_ms"]
            for row in timeline["segments"]
        )
        frames.append(reference_active and not caption_active)
    intervals = []
    start = None
    for index in range(len(frames) + 1):
        absent = frames[index] if index < len(frames) else False
        if absent and start is None:
            start = index * frame_step_ms
        if not absent and start is not None:
            end = index * frame_step_ms
            intervals.append({
                "start_ms": start, "end_ms": end,
                "duration_ms": end - start,
            })
            start = None
    return {
        "gap_count": len(intervals),
        "reference_speech_without_captions_seconds": round(
            sum(i["duration_ms"] for i in intervals) / 1000, 2
        ),
        "longest_gaps": sorted(
            intervals, key=lambda x: -x["duration_ms"],
        )[:24],
        "requires_audio_review": True,
        "source": "human-annotation-derived AMI RTTM versus published captions",
        "note": (
            "Reference speech not covered by any transcript segment. "
            "This does not prove a particular ASR/VAD failure or authorize "
            "automatically inserting unverified words."
        ),
    }


def reference_speaker_coverage(
    timeline: dict, reference: list[dict], duration_ms: int,
    *, frame_step_ms: int = 100,
) -> dict:
    """Audit acoustic evidence per human reference voice, not speaker accuracy.

    Published speaker labels are anonymized and NOT presumed to match AMI
    reference speakers. A caption that temporally overlaps a reference
    speaker may actually belong to a different speaker, especially during
    overlap. The output is for reviewing missed/low-frequency speakers.
    """
    if frame_step_ms != 100 or duration_ms <= 0:
        raise ValueError("Reference speaker audit requires 100ms frames")
    speakers = sorted({turn["speaker"] for turn in reference})
    if not speakers:
        raise ValueError("Empty speaker reference")
    states = {
        speaker: {
            "reference": 0, "no_caption": 0, "unknown_only": 0,
            "positive_caption": 0, "caption_without_embedding": 0,
            "caption_with_embedding": 0, "overlap": 0,
            "contacts": {},
        }
        for speaker in speakers
    }
    for time_ms in range(0, duration_ms, frame_step_ms):
        active_ref = {
            row["speaker"] for row in reference
            if row["start_ms"] <= time_ms < row["end_ms"]
        }
        if not active_ref:
            continue
        live = [
            row for row in timeline["segments"]
            if row["start_ms"] <= time_ms < row["end_ms"]
        ]
        positive = {
            int(row["published_speaker_id"]) for row in live
            if type(row.get("published_speaker_id")) is int
            and row["published_speaker_id"] > 0
        }
        embedded = any(bool(row.get("embedding_available")) for row in live)
        for speaker in active_ref:
            state = states[speaker]
            state["reference"] += 1
            if len(active_ref) >= 2:
                state["overlap"] += 1
            if not live:
                state["no_caption"] += 1
            elif not positive:
                state["unknown_only"] += 1
            else:
                state["positive_caption"] += 1
                for label in positive:
                    state["contacts"][label] = (
                        state["contacts"].get(label, 0) + 1
                    )
            if live and not embedded:
                state["caption_without_embedding"] += 1
            elif embedded:
                state["caption_with_embedding"] += 1

    def seconds(frames: int) -> float:
        return round(frames * frame_step_ms / 1000, 2)

    rows = []
    for speaker in speakers:
        state = states[speaker]
        turns = [item for item in reference if item["speaker"] == speaker]
        if (state["reference"] != state["no_caption"]
                + state["unknown_only"] + state["positive_caption"]):
            raise AssertionError("Reference speaker coverage is inconsistent")
        contacts = sorted(
            state["contacts"].items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        rows.append({
            "reference_speaker": speaker,
            "annotated_intervals": len(turns),
            "short_intervals_under_1_2s": sum(
                item["end_ms"] - item["start_ms"] < 1200
                for item in turns
            ),
            "reference_speech_seconds": seconds(state["reference"]),
            "without_caption_seconds": seconds(state["no_caption"]),
            "unknown_only_caption_seconds": seconds(state["unknown_only"]),
            "positive_caption_seconds": seconds(state["positive_caption"]),
            "caption_without_embedding_seconds": seconds(
                state["caption_without_embedding"]
            ),
            "caption_with_embedding_seconds": seconds(
                state["caption_with_embedding"]
            ),
            "overlap_reference_seconds": seconds(state["overlap"]),
            "positive_label_contacts": [
                {
                    "published_label": label,
                    "temporal_overlap_seconds": seconds(frames),
                }
                for label, frames in contacts
            ],
        })
    rows.sort(
        key=lambda item: (-item["reference_speech_seconds"],
                          item["reference_speaker"])
    )
    return {
        "frame_step_ms": frame_step_ms,
        "reference_speaker_count": len(rows),
        "speakers": rows,
        "interpretation": (
            "Temporal caption and anonymous label coverage per AMI voice, "
            "NOT identity correctness. One live label can contact multiple "
            "reference speakers; simultaneous speech is counted for each "
            "active speaker. RTTM only_words intervals are not necessarily "
            "complete conversational turns. No recognized utterances are exported."
        ),
    }


def reference_speaker_context_breakdown(
    timeline: dict, reference: list[dict], duration_ms: int,
    *, frame_step_ms: int = 100,
) -> dict:
    """Separate isolated reference speech from overlapping human speech.

    Compare only temporal caption/embedding coverage, not inferred IDs.
    Return the longest solo stretches for targeted original-WAV listening.
    """
    if frame_step_ms != 100 or duration_ms <= 0:
        raise ValueError("Context breakdown requires 100ms frames")
    names = sorted({item["speaker"] for item in reference})
    records: dict[str, dict] = {}
    for speaker in names:
        records[speaker] = {}
        for context in ("solo", "overlap"):
            records[speaker][context] = {
                "total_frames": 0,
                "no_caption_frames": 0,
                "unknown_only_frames": 0,
                "positive_caption_frames": 0,
                "without_embedding_frames": 0,
                "with_embedding_frames": 0,
                "label_contacts": {},
            }
        records[speaker]["solo_runs"] = []
        records[speaker]["solo_start"] = None

    for ms in range(0, duration_ms + frame_step_ms, frame_step_ms):
        active = set() if ms >= duration_ms else {
            item["speaker"] for item in reference
            if item["start_ms"] <= ms < item["end_ms"]
        }
        visible = [] if ms >= duration_ms else [
            item for item in timeline["segments"]
            if item["start_ms"] <= ms < item["end_ms"]
        ]
        labels = {
            item["published_speaker_id"] for item in visible
            if type(item.get("published_speaker_id")) is int
            and item["published_speaker_id"] > 0
        }
        with_embedding = any(
            bool(item.get("embedding_available")) for item in visible
        )
        for speaker in names:
            state = records[speaker]
            is_solo = speaker in active and len(active) == 1
            if is_solo and state["solo_start"] is None:
                state["solo_start"] = ms
            if not is_solo and state["solo_start"] is not None:
                begin = state["solo_start"]
                if ms > begin:
                    state["solo_runs"].append({
                        "start_ms": begin, "end_ms": ms,
                        "duration_ms": ms - begin,
                    })
                state["solo_start"] = None
            if speaker not in active:
                continue
            context = "solo" if is_solo else "overlap"
            values = state[context]
            values["total_frames"] += 1
            if not visible:
                values["no_caption_frames"] += 1
            elif not labels:
                values["unknown_only_frames"] += 1
            else:
                values["positive_caption_frames"] += 1
                for label in labels:
                    contacts = values["label_contacts"]
                    contacts[label] = contacts.get(label, 0) + 1
            if visible and not with_embedding:
                values["without_embedding_frames"] += 1
            elif with_embedding:
                values["with_embedding_frames"] += 1

    def format_context(values: dict) -> dict:
        if (values["total_frames"] != values["no_caption_frames"]
                + values["unknown_only_frames"]
                + values["positive_caption_frames"]):
            raise AssertionError("Context speech time does not reconcile")
        return {
            "reference_speech_seconds": round(
                values["total_frames"] * frame_step_ms / 1000, 2
            ),
            "without_caption_seconds": round(
                values["no_caption_frames"] * frame_step_ms / 1000, 2
            ),
            "unknown_only_caption_seconds": round(
                values["unknown_only_frames"] * frame_step_ms / 1000, 2
            ),
            "positive_caption_seconds": round(
                values["positive_caption_frames"] * frame_step_ms / 1000, 2
            ),
            "caption_without_embedding_seconds": round(
                values["without_embedding_frames"] * frame_step_ms / 1000, 2
            ),
            "caption_with_embedding_seconds": round(
                values["with_embedding_frames"] * frame_step_ms / 1000, 2
            ),
            "positive_label_contacts": [
                {
                    "published_label": label,
                    "temporal_overlap_seconds": round(
                        frames * frame_step_ms / 1000, 2
                    ),
                }
                for label, frames in sorted(
                    values["label_contacts"].items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )
            ],
        }

    result = []
    for speaker in names:
        value = records[speaker]
        solo = format_context(value["solo"])
        overlap = format_context(value["overlap"])
        result.append({
            "reference_speaker": speaker,
            "solo": solo, "overlap": overlap,
            "longest_solo_stretches": sorted(
                value["solo_runs"],
                key=lambda item: (-item["duration_ms"], item["start_ms"]),
            )[:12],
        })
    result.sort(
        key=lambda item: (
            -(item["solo"]["reference_speech_seconds"]
              + item["overlap"]["reference_speech_seconds"]),
            item["reference_speaker"],
        )
    )
    return {
        "frame_step_ms": frame_step_ms,
        "speakers": result,
        "interpretation": (
            "Solo means exactly one AMI reference speaker is active. "
            "Positive caption label contact is only temporal evidence, not "
            "proof of speaker identity. Solo stretches are 100ms-quantized "
            "review intervals, not automatic trusted enrollment clips."
        ),
    }


def score_case(root: Path, meeting: str, replay_path: Path) -> dict:
    root = root.resolve()
    replay_path = replay_path.resolve()
    if (meeting not in {m for m, _ in SELECTED}
            or not replay_path.is_file()
            or not replay_path.is_relative_to(root)):
        raise ValueError("Use a pinned AMI meeting and private local replay JSON")
    path = root / "cases" / (meeting + ".json")
    if not path.is_file():
        raise FileNotFoundError("Prepared AMI case missing; run ami_shadow10_prepare")
    case = json.loads(path.read_text(encoding="utf-8"))
    expected_offset = dict(SELECTED)[meeting]
    if (case.get("source_meeting") != meeting
            or case.get("start_seconds") != expected_offset
            or case.get("end_seconds") != expected_offset + DURATION_S):
        raise ValueError("AMI case provenance mismatch")
    relative = case.get("audio")
    if not isinstance(relative, str):
        raise ValueError("Missing local AMI WAV path")
    audio = (root / relative).resolve()
    if (not audio.is_relative_to(root) or not audio.is_file()
            or audio.suffix.lower() != ".wav"
            or checksum(audio) != case.get("sha256")):
        raise ValueError("AMI source WAV missing or checksum changed")
    samples = load_wav(audio)
    duration_ms = round(len(samples) * 1000 / 16000)
    if duration_ms != DURATION_S * 1000:
        raise ValueError("AMI excerpt must be exactly 240 seconds")
    # Never trust a case manifest's embedded reference labels alone.
    # Re-derive them from the pinned local RTTM and verify original bytes.
    rttm = root / "rttm" / (meeting + ".rttm")
    if (not rttm.is_file()
            or checksum(rttm) != case.get("reference_source_sha256")):
        raise ValueError("AMI RTTM missing or reference checksum changed")
    reference = _reference_turns(case.get("reference_turns"), duration_ms)
    frozen = parse_reference(
        rttm.read_text(encoding="utf-8"), meeting, expected_offset,
    )
    if frozen != reference:
        raise ValueError("AMI annotated reference does not match source RTTM")
    validate_reference(reference)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    source_name = str(replay.get("source_audio") or "").replace("\\", "/").split("/")[-1]
    if (source_name != audio.name
            or replay.get("source_audio_sha256") != case["sha256"]
            or replay.get("loops") != 1 or replay.get("playback_speed") != 1.0):
        raise ValueError("Replay is not the same single-pass 1x frozen AMI WAV")
    timeline = (replay.get("final_cluster_diagnostics") or {}).get(
        "anonymous_timeline"
    )
    if not isinstance(timeline, dict):
        raise ValueError(
            "Missing anonymous timeline. Restart backend with "
            "MEETING_DIARIZATION_DIAGNOSTICS=1 and replay into a new file."
        )
    predictions = baseline_turns(replay, duration_ms=duration_ms)
    if (type(timeline.get("total_segments")) is not int
            or timeline["total_segments"] != replay.get("final_confirmed_segments")):
        raise ValueError("Published timeline count differs from final transcript")
    unknown = unidentified_segments(timeline, reference, duration_ms)
    per_speaker = reference_speaker_coverage(
        timeline, reference, duration_ms,
    )
    context_breakdown = reference_speaker_context_breakdown(
        timeline, reference, duration_ms,
    )
    caption_gaps = reference_speech_without_captions(
        timeline, reference, duration_ms,
    )
    expected_unknown = replay.get("final_unidentified_segment_count")
    if type(expected_unknown) is not int or unknown["count"] != expected_unknown:
        raise ValueError("Unidentified timeline differs from final replay")
    metrics = _score_diarization(reference, predictions)
    missed_breakdown = missed_speaker_time_breakdown(
        timeline, reference, duration_ms,
    )
    reference_seconds = missed_breakdown["reference_speaker_seconds"]
    if (reference_seconds
            and abs(
                missed_breakdown["total_missed_speaker_seconds"] / reference_seconds
                - metrics["missed_speaker_time"]
            ) > 0.002):
        raise ValueError("AMI missed-time decomposition does not match DER")
    return {
        "schema": SCHEMA,
        "meeting": meeting,
        "source_audio_sha256": case["sha256"],
        "replay_sha256": checksum(replay_path),
        "reference_source_sha256": case.get("reference_source_sha256"),
        "reference_kind": case.get("reference_kind"),
        "timeline_complete": True,
        "frame_step_ms": metrics["frame_step_ms"],
        "collar_ms": 0,
        "metrics": metrics,
        "reference_speaker_coverage": per_speaker,
        "reference_speaker_context": context_breakdown,
        "missed_speaker_time_breakdown": missed_breakdown,
        "unidentified": unknown,
        "reference_speech_without_captions": caption_gaps,
        "confirmed_segment_count": timeline["total_segments"],
        "shadow_diagnostics": replay.get("speaker_shadow_diagnostics"),
        "limitations": (
            "Human-derived AMI only_words speaker spans, not all vocal events. "
            "100ms no-collar frame scorer with overlap; live CAM++ "
            "ASR-aligned segments cannot be compared to offline models "
            "without a paired reference. Temporal overlap of an unidentified "
            "segment is diagnostic, not a verified error cause."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark/fixed195/ami_shadow10")
    )
    parser.add_argument("--meeting", required=True, choices=[x[0] for x in SELECTED])
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    private = Path("benchmark/fixed195").resolve()
    output = args.output.resolve()
    if (not root.is_relative_to(private)
            or not output.is_relative_to(root)
            or output.suffix.lower() != ".json"):
        parser.error("The AMI root and output JSON must stay in gitignored Fixed195")
    if output.exists() or output == args.replay.resolve():
        parser.error("Refusing to overwrite a replay or existing result")
    result = score_case(root, args.meeting, args.replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as dest:
        json.dump(result, dest, indent=2)
        dest.write("\n")
    summary = {
        "meeting": result["meeting"],
        "metrics": result["metrics"],
        "speaker_coverage": result["reference_speaker_coverage"]["speakers"],
        "missed_speaker_time_breakdown": result["missed_speaker_time_breakdown"],
        "unidentified_count": result["unidentified"]["count"],
        "reference_speech_without_captions_seconds": (
            result["reference_speech_without_captions"][
                "reference_speech_without_captions_seconds"
            ]
        ),
        "unidentified_with_reference_speech":
            result["unidentified"]["with_reference_speech"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

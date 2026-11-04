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

from tools.ami_shadow10_prepare import DURATION_S, SELECTED, checksum
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
    reference = _reference_turns(case.get("reference_turns"), duration_ms)
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
    expected_unknown = replay.get("final_unidentified_segment_count")
    if type(expected_unknown) is not int or unknown["count"] != expected_unknown:
        raise ValueError("Unidentified timeline differs from final replay")
    metrics = _score_diarization(reference, predictions)
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
        "unidentified": unknown,
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
        "unidentified_count": result["unidentified"]["count"],
        "unidentified_with_reference_speech":
            result["unidentified"]["with_reference_speech"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

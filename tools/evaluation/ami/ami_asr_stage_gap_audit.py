"""Locate why AMI reference speech has no final caption.

Per-frame comparison of frozen human RTTM against observed acoustic VAD,
raw Whisper timestamps, quality-verified ASR and confirmed captions.
This is evidence of timestamp coverage only, NOT a word-error audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech.captions.asr_stage_trace import merged_intervals
from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.evaluation.speakers.speaker_stage1_compare import _reference_turns

STAGES = (
    "vad_intervals_ms", "raw_model_intervals_ms",
    "verified_intervals_ms", "after_recovery_intervals_ms",
)


def audit_stages(
    root: Path, meeting: str, replay_path: Path, *,
    reference_protocol: str = "only_words",
) -> dict:
    if reference_protocol not in ("only_words", "forced_aligned"):
        raise ValueError("Unsupported reference protocol")
    verified = score_case(root, meeting, replay_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    trace = replay.get("asr_coverage_trace")
    if (not isinstance(trace, dict)
            or trace.get("schema") != "meeting-asr-stage-trace-v1"
            or trace.get("truncated") is not False
            or not isinstance(trace.get("cycles"), list)
            or not trace["cycles"]):
        raise ValueError(
            "Complete ASR stage trace missing. Start backend with "
            "MEETING_ASR_TRACE_STAGES=1 and replay into a new file."
        )
    duration_ms = round(replay["source_audio_duration_s"] * 1000)
    if duration_ms != 240000:
        raise ValueError("Expected frozen 240 s AMI excerpt")
    if reference_protocol == "only_words":
        reference = _reference_turns(json.loads(
            (root / "cases" / (meeting + ".json")).read_text(
                encoding="utf-8",
            )
        )["reference_turns"], duration_ms)
        reference_hash = verified["reference_source_sha256"]
    else:
        # Use an explicitly different, pinned annotation. Never silently
        # change the original only_words benchmark used for acceptance.
        from tools.evaluation.ami.ami_nemotron_reference_diagnostic import (
            fetch_forced_rttm, forced_reference,
        )
        raw, reference_hash, _ = fetch_forced_rttm(root, meeting)
        reference = forced_reference(raw, meeting, dict(SELECTED)[meeting])
    all_intervals = {}
    for key in STAGES:
        raw = []
        for cycle in trace["cycles"]:
            for span in cycle.get(key, []):
                raw.append(span)
        all_intervals[key] = merged_intervals(raw, duration_ms=duration_ms)
    caption_rows = (
        replay["final_cluster_diagnostics"]["anonymous_timeline"]["segments"]
    )
    confirmed = merged_intervals(
        ((row["start_ms"], row["end_ms"]) for row in caption_rows),
        duration_ms=duration_ms,
    )

    def active(spans, t):
        return any(a <= t < b for a, b in spans)

    counts = {
        "not_detected_by_recorded_vad": 0,
        "vad_without_raw_model_timestamp": 0,
        "raw_model_without_verified_caption": 0,
        "verified_but_not_in_final_caption": 0,
        "verified_only_after_gap_recovery": 0,
        "recovered_but_not_in_final_caption": 0,
        "final_caption_covered": 0,
    }
    missing_intervals = []
    current_start = None
    for t in range(0, duration_ms + 100, 100):
        if t == duration_ms:
            reference_active = False
        else:
            reference_active = any(
                row["start_ms"] <= t < row["end_ms"] for row in reference
            )
        final_visible = active(confirmed, t)
        if reference_active and not final_visible:
            if current_start is None:
                current_start = t
        elif current_start is not None:
            missing_intervals.append({
                "start_ms": current_start, "end_ms": t,
                "duration_ms": t - current_start,
            })
            current_start = None

        if not reference_active:
            continue
        if final_visible:
            counts["final_caption_covered"] += 1
        elif not active(all_intervals["vad_intervals_ms"], t):
            counts["not_detected_by_recorded_vad"] += 1
        elif not active(all_intervals["raw_model_intervals_ms"], t):
            counts["vad_without_raw_model_timestamp"] += 1
        elif not active(all_intervals["verified_intervals_ms"], t):
            if active(all_intervals["after_recovery_intervals_ms"], t):
                counts["verified_only_after_gap_recovery"] += 1
            else:
                counts["raw_model_without_verified_caption"] += 1
        elif active(all_intervals["after_recovery_intervals_ms"], t):
            counts["verified_but_not_in_final_caption"] += 1
        else:
            counts["recovered_but_not_in_final_caption"] += 1
    total_missing = sum(value for key, value in counts.items()
                        if key != "final_caption_covered")
    measured_missing = round(sum(
        item["duration_ms"] for item in missing_intervals
    ) / 1000, 2)
    recorded_missing = verified[
        "reference_speech_without_captions"
    ]["reference_speech_without_captions_seconds"]
    if (reference_protocol == "only_words"
            and abs(measured_missing - recorded_missing) > 0.15):
        raise ValueError("ASR stage audit disagrees with caption reference")
    recoveries = {
        key: sum(max(0, int(cycle.get(key) or 0)) for cycle in trace["cycles"])
        for key in ("recovery_attempted", "recovery_confirmed", "recovery_failed")
    }
    receipts = [
        cycle["commit_receipt"] for cycle in trace["cycles"]
        if isinstance(cycle.get("commit_receipt"), dict)
    ]
    receipt_fields = (
        "verified_candidates", "held", "invalid_timestamps",
        "duplicate_skipped", "published_new", "published_from_recovery",
    )
    commit_totals = (
        {name: sum(int(row.get(name, 0)) for row in receipts)
         for name in receipt_fields}
        if receipts else None
    )
    if commit_totals is not None and (
        commit_totals["verified_candidates"] != sum(
            commit_totals[key] for key in (
                "held", "invalid_timestamps", "duplicate_skipped",
                "published_new",
            )
        )
    ):
        raise ValueError("ASR commit receipt conservation failed")
    bypassed_cycles = sum(bool(cycle.get("window_internal_vad_bypassed"))
                          for cycle in trace["cycles"])
    return {
        "schema": "meeting-ami-asr-stage-gap-audit-v1",
        "meeting": meeting,
        "source_audio_sha256": verified["source_audio_sha256"],
        "reference_source_sha256": reference_hash,
        "reference_protocol": reference_protocol,
        "independent_reference_protocol_from_original": (
            reference_protocol != "only_words"
        ),
        "window_internal_vad_bypassed_cycles": bypassed_cycles,
        "recovery_counts": recoveries,
        "commit_totals_across_cycles": commit_totals,
        "commit_receipt_cycles": len(receipts),
        "commit_count_note": (
            "Candidate counts are per ASR inference cycle and may include "
            "the same audio on multiple passes. Published_new is the count "
            "of new canonical segments at the point of initial commit, "
            "not word correctness, WER, or later delivery guarantee."
        ),
        "reference_clock_no_caption_seconds": measured_missing,
        "reference_clock_covered_seconds": counts["final_caption_covered"] / 10,
        "gap_causes_seconds": {
            key: round(value / 10, 2) for key, value in counts.items()
            if key != "final_caption_covered"
        },
        "longest_no_caption_gaps": sorted(
            missing_intervals, key=lambda x: -x["duration_ms"]
        )[:24],
        "trace_cycles": len(trace["cycles"]),
        "explanation": (
            "Classified by the earliest observed missing timestamp stage, "
            "using a union across all ASR retries. These are not ground-truth "
            "words or proof of one model failure. Confirm with source audio "
            "and separate verified word transcripts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "benchmark/fixed195/ami_shadow10",
    ))
    parser.add_argument("--meeting", choices=[m for m, _ in SELECTED],
                        required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-protocol", default="only_words",
        choices=("only_words", "forced_aligned"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.output.resolve()
    if (not root.is_relative_to(Path("benchmark/fixed195").resolve())
            or not path.is_relative_to(root)
            or path.suffix != ".json" or path.exists()):
        parser.error("Choose a NEW private JSON inside the Fixed195 root")
    report = audit_stages(
        root, args.meeting, args.replay,
        reference_protocol=args.reference_protocol,
    )
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare two AMI replays frame-by-frame without changing the benchmark.

Reports net, newly gained and newly LOST reference-speech coverage. A net
increase alone hides churn from nondeterministic ASR windows. The diagnostic
contains no decoded text and cannot certify recovered word correctness.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from speech.asr_stage_trace import merged_intervals
from tools.ami_shadow10_prepare import SELECTED
from tools.ami_shadow10_score import score_case
from tools.ami_nemotron_reference_diagnostic import (
    fetch_forced_rttm, forced_reference,
)
from tools.speaker_stage1_compare import _reference_turns

DURATION_MS = 240000
STEP_MS = 100


def _caption_intervals(replay: dict) -> list[list[int]]:
    timeline = (
        replay.get("final_cluster_diagnostics") or {}
    ).get("anonymous_timeline")
    if (not isinstance(timeline, dict)
            or timeline.get("truncated") is not False):
        raise ValueError("Full final caption timeline required")
    rows = timeline.get("segments")
    if (not isinstance(rows, list)
            or timeline.get("total_segments") != len(rows)):
        raise ValueError("Truncated or inconsistent caption timeline")
    intervals = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Invalid caption record")
        a, b = item.get("start_ms"), item.get("end_ms")
        if (type(a) is not int or type(b) is not int
                or a < 0 or b <= a or b > DURATION_MS):
            raise ValueError("Invalid caption times")
        intervals.append((a, b))
    return merged_intervals(intervals, duration_ms=DURATION_MS)


def framewise_caption_change(
    reference: list[dict], baseline: list[list[int]],
    candidate: list[list[int]], *, duration_ms: int = DURATION_MS,
) -> dict:
    """Score changes in reference-active clock frames, never text strings.

    Reference active means at least one true speaker is active; overlap
    contributes one CLOCK frame, not multiple reference speaker-seconds.
    """
    if duration_ms < 1 or duration_ms % STEP_MS:
        raise ValueError("Frame grid must be a positive multiple of 100ms")
    counts = {
        "reference_frames": 0,
        "baseline_covered": 0,
        "candidate_covered": 0,
        "newly_covered": 0,
        "coverage_lost": 0,
        "covered_both": 0,
        "missing_both": 0,
    }
    for t in range(0, duration_ms, STEP_MS):
        active = any(
            row["start_ms"] <= t < row["end_ms"] for row in reference
        )
        if not active:
            continue
        before = any(a <= t < b for a, b in baseline)
        after = any(a <= t < b for a, b in candidate)
        counts["reference_frames"] += 1
        counts["baseline_covered"] += int(before)
        counts["candidate_covered"] += int(after)
        counts["newly_covered"] += int(after and not before)
        counts["coverage_lost"] += int(before and not after)
        counts["covered_both"] += int(before and after)
        counts["missing_both"] += int(not before and not after)
    return {
        "reference_clock_s": counts["reference_frames"] / 10,
        "baseline_covered_s": counts["baseline_covered"] / 10,
        "candidate_covered_s": counts["candidate_covered"] / 10,
        "newly_covered_s": counts["newly_covered"] / 10,
        "coverage_lost_s": counts["coverage_lost"] / 10,
        "unchanged_covered_s": counts["covered_both"] / 10,
        "unchanged_missing_s": counts["missing_both"] / 10,
        "net_new_coverage_s": round(
            (counts["newly_covered"] - counts["coverage_lost"]) / 10, 2,
        ),
    }


def paired_report(
    root: Path, meeting: str, baseline_path: Path, candidate_path: Path,
    *, protocol: str = "forced_aligned",
) -> dict:
    if meeting not in dict(SELECTED):
        raise ValueError("Not a frozen AMI meeting")
    if protocol not in ("only_words", "forced_aligned"):
        raise ValueError("Unsupported reference protocol")
    for path in (baseline_path, candidate_path):
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Replay is outside the private AMI benchmark root")
    original_a = score_case(root, meeting, baseline_path)
    original_b = score_case(root, meeting, candidate_path)
    if (original_a["source_audio_sha256"] != original_b["source_audio_sha256"]
            or original_a["reference_source_sha256"]
            != original_b["reference_source_sha256"]):
        raise ValueError("Replay audio or frozen reference changed")
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (before.get("source_audio_duration_s") != 240.0
            or after.get("source_audio_duration_s") != 240.0):
        raise ValueError("Both replays must be complete 240-second excerpts")
    if protocol == "forced_aligned":
        raw, digest, _ = fetch_forced_rttm(root, meeting)
        reference = forced_reference(raw, meeting, dict(SELECTED)[meeting])
    else:
        digest = original_a["reference_source_sha256"]
        case = json.loads(
            (root / "cases" / (meeting + ".json")).read_text(encoding="utf-8")
        )
        reference = _reference_turns(
            case["reference_turns"], DURATION_MS,
        )
    change = framewise_caption_change(
        reference, _caption_intervals(before), _caption_intervals(after),
    )
    cycle_trace = after.get("asr_coverage_trace") or {}
    cycles = cycle_trace.get("cycles")
    if (not isinstance(cycles, list)
            or cycle_trace.get("truncated") is not False):
        cycles = None
    timeline = after["final_cluster_diagnostics"]["anonymous_timeline"]
    final_rows = timeline["segments"]
    has_origin = all("recovery_origin" in row for row in final_rows)
    published_recovered_ids = (
        len({row["segment_id"] for row in final_rows
             if row.get("recovery_origin") is True})
        if has_origin else None
    )
    return {
        "schema": "meeting-ami-paired-caption-coverage-v1",
        "meeting": meeting,
        "reference_protocol": protocol,
        "reference_sha256": digest,
        "source_audio_sha256": original_a["source_audio_sha256"],
        "baseline_replay_sha256": sha256(baseline_path.read_bytes()).hexdigest(),
        "candidate_replay_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
        "coverage": change,
        "candidate_recovery_attempts_across_cycles": (
            sum(int(c.get("recovery_attempted") or 0) for c in cycles)
            if cycles is not None else None
        ),
        "candidate_recovery_confirmations_across_cycles": (
            sum(int(c.get("recovery_confirmed") or 0) for c in cycles)
            if cycles is not None else None
        ),
        "candidate_distinct_published_recovery_segments": published_recovered_ids,
        "note": (
            "Counts are reference-speech CLOCK seconds on 100ms frames. "
            "Gain and loss may coexist; net gain alone hides churn. "
            "Retry confirmations are per-cycle candidates; published "
            "recovery segments are distinct final segment IDs only when "
            "explicit provenance is present (null for older replays). "
            "Neither count is verified recovered words or CER. "
            "This tool does not alter the frozen only_words DER gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "benchmark/fixed195/ami_shadow10",
    ))
    parser.add_argument("--meeting", default="TS3003b")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--reference-protocol",
        default="forced_aligned", choices=("only_words", "forced_aligned"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if (not root.is_relative_to(Path("benchmark/fixed195").resolve())
            or not output.is_relative_to(root)
            or output.exists() or output.suffix != ".json"):
        parser.error("New JSON output must be inside private Fixed195 root")
    report = paired_report(
        root, args.meeting, args.baseline.resolve(), args.candidate.resolve(),
        protocol=args.reference_protocol,
    )
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

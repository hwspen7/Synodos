"""Analyze frozen AMI DER failure patterns without rerunning CUDA models.

Reference overlap pressure is a property of human annotations. It is NOT
proof that any particular missed speaker-second was caused by overlap.
No individual speaker ID, voiceprint, transcript or audio samples are
written into this aggregate diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from tools.ami_shadow10_prepare import DURATION_S, SELECTED
from tools.phase1b_pyannote_ami10 import summarize, validate_report

STEP_MS = 100
METRIC_KEYS = (
    "der", "missed_speaker_time", "false_alarm_time", "confusion_time"
)


def reference_pressure(turns: list[dict]) -> dict:
    """Measure overlapping speaker occupancy using the exact DER frame grid."""
    if not isinstance(turns, list) or not turns:
        raise ValueError("Human-reference speaker intervals are required")
    duration_ms = DURATION_S * 1000
    total_frames = duration_ms // STEP_MS
    speech_frames = overlap_frames = speaker_frames = extra_frames = 0
    speakers = set()
    for row in turns:
        a, b, speaker = (row.get("start_ms"), row.get("end_ms"),
                         row.get("speaker"))
        if (type(a) is not int or type(b) is not int
                or a < 0 or b <= a or b > duration_ms
                or not isinstance(speaker, str) or not speaker):
            raise ValueError("Malformed frozen human speaker interval")
        speakers.add(speaker)
    for i in range(total_frames):
        now = i * STEP_MS
        active = {
            row["speaker"] for row in turns
            if row["start_ms"] <= now < row["end_ms"]
        }
        n = len(active)
        speech_frames += n > 0
        overlap_frames += n >= 2
        speaker_frames += n
        extra_frames += max(0, n - 1)

    return {
        "reference_speaker_count": len(speakers),
        "reference_speaker_seconds": round(speaker_frames * STEP_MS / 1000, 2),
        "reference_speech_wall_seconds": round(speech_frames * STEP_MS / 1000, 2),
        "reference_overlap_wall_seconds": round(overlap_frames * STEP_MS / 1000, 2),
        "reference_overlap_extra_speaker_seconds": round(extra_frames * STEP_MS / 1000, 2),
        "reference_overlap_fraction_of_speech_wall": round(
            overlap_frames / speech_frames, 5
        ) if speech_frames else None,
        "reference_extra_speaker_fraction": round(
            extra_frames / speaker_frames, 5
        ) if speaker_frames else None,
    }


def diagnose(reports: list[dict], cases: dict[str, dict], root: Path) -> dict:
    """Join immutable real-model metrics with reference-only overlap pressure."""
    scored = summarize(reports, root)
    if set(cases) != {name for name, _ in SELECTED}:
        raise ValueError("Exactly ten frozen reference cases are required")
    lookup = {record["meeting"]: record for record in reports}
    rows = []
    for meeting, _ in SELECTED:
        report = validate_report(lookup[meeting], root, meeting)
        metrics = report["metrics"]
        pressure = reference_pressure(cases[meeting]["reference_turns"])
        if abs(
            pressure["reference_speaker_seconds"]
            - metrics["reference_speaker_seconds"]
        ) > 0.011:
            raise ValueError(f"{meeting}: reference speaker time differs from DER grid")
        diag = report.get("diagnostics") or {}
        count = diag.get("predicted_speakers")
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError(f"{meeting}: invalid predicted speaker count")
        rows.append({
            "meeting": meeting,
            "source_group": meeting[:-1],
            "der": metrics["der"],
            "missed_speaker_time": metrics["missed_speaker_time"],
            "false_alarm_time": metrics["false_alarm_time"],
            "confusion_time": metrics["confusion_time"],
            "reference_speaker_seconds": metrics["reference_speaker_seconds"],
            "predicted_speaker_count": count,
            **pressure,
        })
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    group_summaries = []
    for group, entries in sorted(groups.items()):
        seconds = sum(e["reference_speaker_seconds"] for e in entries)
        weighted = {
            key: round(sum(e[key] * e["reference_speaker_seconds"]
                            for e in entries) / seconds, 5)
            for key in METRIC_KEYS
        }
        group_summaries.append({
            "source_group": group,
            "meeting_count": len(entries),
            "weighted_metrics": weighted,
            "reference_speaker_seconds": round(seconds, 2),
            "reference_overlap_extra_speaker_seconds": round(sum(
                e["reference_overlap_extra_speaker_seconds"] for e in entries
            ), 2),
        })
    return {
        "schema": "meeting-phase1b-pyannote-reference-pressure-v1",
        "meeting_count": len(rows),
        "overall_weighted_metrics": scored["weighted_metrics"],
        "per_meeting": rows,
        "per_source_group": group_summaries,
        "method": (
            "Frozen AMI only_words references, 100ms frame-start sampling, "
            "zero collar; independently measured overlap plus saved real "
            "Pyannote DER reports"
        ),
        "interpretation_limits": [
            "Overlap reference time does not identify which missed intervals are overlap errors.",
            "DER miss counts missing speaker capacity on reference-active frames, including overlap.",
            "Predicted speaker count is a global count, not proof of correctly recovered identity.",
            "Comparing causality needs time-aligned predicted turns or a new inference.",
            "No production model selection follows from this diagnostic.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    private = Path("benchmark/fixed195").resolve()
    root = private / "ami_shadow10"
    summary_file = args.summary.resolve(strict=True)
    if (summary_file.name != "pyannote_ami10_summary.json"
            or not summary_file.is_relative_to(root / "phase1b")):
        parser.error("Supply the actual private AMI10 Pyannote cohort summary")
    folder = summary_file.parent
    result_file = (args.output or folder / "reference_overlap_diagnostic.json").resolve()
    if (not result_file.is_relative_to(private) or result_file.suffix != ".json"
            or result_file.exists()):
        parser.error("Write a new private JSON report without overwriting old evidence")
    original = json.loads(summary_file.read_text(encoding="utf-8"))
    if (original.get("schema") != "meeting-phase1b-pyannote-ami10-cohort-v1"
            or original.get("meeting_count") != len(SELECTED)):
        raise ValueError("Invalid ten-meeting cohort summary")
    cases = {}
    reports = []
    for meeting, _ in SELECTED:
        path = root / "cases" / f"{meeting}.json"
        cases[meeting] = json.loads(path.read_text(encoding="utf-8"))
        reports.append(json.loads(
            (folder / f"pyannote-{meeting}.json").read_text(encoding="utf-8")
        ))
    result = diagnose(reports, cases, root)
    for key in METRIC_KEYS:
        if abs(result["overall_weighted_metrics"][key]
               - original["weighted_metrics"][key]) > 0.00002:
            raise ValueError("Cohort source reports disagree with summary")
    with result_file.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=True, allow_nan=False)
        out.write("\n")
    print("Meeting  DER    Missed  Overlap-extra  RefSpk  PredSpk")
    for row in sorted(result["per_meeting"], key=lambda r: -r["der"]):
        predicted = (str(row["predicted_speaker_count"])
                     if row["predicted_speaker_count"] is not None else "?")
        print(f"{row['meeting']:<8} {row['der']*100:>5.2f}% "
              f"{row['missed_speaker_time']*100:>6.2f}% "
              f"{row['reference_extra_speaker_fraction']*100:>12.2f}% "
              f"{row['reference_speaker_count']:>6} {predicted:>8}")
    print("\nSource group  Weighted DER  Weighted missed")
    for g in sorted(result["per_source_group"], key=lambda x:
                    -x["weighted_metrics"]["der"]):
        print(f"{g['source_group']:<12} "
              f"{g['weighted_metrics']['der']*100:>10.2f}% "
              f"{g['weighted_metrics']['missed_speaker_time']*100:>15.2f}%")
    print("Overlap-extra is reference-only; it is NOT attributed model error.")
    print("Diagnostic:", result_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

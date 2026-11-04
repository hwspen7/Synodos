"""Strict held-out AMI evidence gate for candidate speech pipelines.

The standard non-regression guard tolerates small losses. This gate is
intentionally stronger: a quality *gain* must be measured, not inferred
from a green test suite or a different speaker-count heuristic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.ami_live_regression_gate import compare_live_audits


def assess_quality_gain(baseline: dict, candidate: dict) -> dict:
    paired = compare_live_audits(baseline, candidate)
    measurements = paired["measurements"]
    base_metrics = baseline["metrics"]
    new_metrics = candidate["metrics"]
    for label, metrics in (("baseline", base_metrics), ("candidate", new_metrics)):
        for key in ("speaker_count_absolute_error", "overlap_all_speakers_recall"):
            if key not in metrics:
                raise ValueError(f"{label} missing speaker-count/overlap metric: {key}")
    strict = {
        "paired_regression_gate_passed": paired["non_regression_pass"],
        "der_improved_by_at_least_0p2pp":
            measurements["der"]["delta"] <= -0.002,
        "missed_speaker_time_not_worse":
            measurements["missed_speaker_time"]["delta"] <= 0,
        "confusion_time_not_worse":
            measurements["confusion_time"]["delta"] <= 0,
        "false_alarm_not_worse_by_over_0p1pp":
            measurements["false_alarm_time"]["delta"] <= 0.001,
        "no_caption_clock_improved_by_0p5s":
            measurements["no_caption_clock_seconds"]["delta"] <= -0.5,
        "without_transcript_speaker_time_not_worse":
            measurements["without_transcript_speaker_seconds"]["delta"] <= 0,
        "speaker_count_error_not_worse":
            new_metrics["speaker_count_absolute_error"]
            <= base_metrics["speaker_count_absolute_error"],
        "overlap_recall_not_worse":
            new_metrics["overlap_all_speakers_recall"]
            >= base_metrics["overlap_all_speakers_recall"],
    }
    return {
        "schema": "meeting-ami-strict-quality-gain-v1",
        "meeting": baseline["meeting"],
        "measured_gain_on_this_case": all(strict.values()),
        "checks": strict,
        "changes": measurements,
        "speaker_count_error": {
            "baseline": base_metrics["speaker_count_absolute_error"],
            "candidate": new_metrics["speaker_count_absolute_error"],
        },
        "overlap_all_speakers_recall": {
            "baseline": base_metrics["overlap_all_speakers_recall"],
            "candidate": new_metrics["overlap_all_speakers_recall"],
        },
        "limits": (
            "Single-case improvement is not a production promotion. "
            "Repeat on multiple AMI clips and independently checked "
            "multilingual WER/CER, especially short turns and overlap. "
            "A candidate with missing or incompatible provenance is rejected."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    args = p.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = assess_quality_gain(baseline, candidate)
    print(json.dumps(result, indent=2))
    return 0 if result["measured_gain_on_this_case"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare strictly paired Phase 1B AMI candidate reports.

This statistical summary is NOT a new DER result. Only real model reports
with identical audio/reference/case hashes can be aggregated. A report set
lacking a complete held-out meeting cohort cannot approve a model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.ami_shadow10_prepare import SELECTED

MODELS = ("nemotron3", "pyannote", "diarizen")


def compare_reports(reports: list[dict], *, min_meetings: int = 10) -> dict:
    if not 1 <= min_meetings <= len(SELECTED):
        raise ValueError("Invalid meeting cohort size")
    cases = {}
    for record in reports:
        if record.get("schema") != "meeting-phase1b-frozen-diarization-v1":
            raise ValueError("Unknown or unpinned Phase 1B report schema")
        model, meeting = record.get("model"), record.get("meeting")
        if model not in MODELS or meeting not in dict(SELECTED):
            raise ValueError("Unrecognized model or AMI meeting")
        key = (meeting, model)
        if key in cases:
            raise ValueError("Duplicate model evaluation for one frozen meeting")
        if (record.get("frame_step_ms") != 100
                or record.get("collar_ms") != 0
                or record.get("overlap_included") is not True
                or record.get("reference_protocol") != "only_words"):
            raise ValueError("Diarization scoring protocol changed")
        for field in ("source_audio_sha256", "reference_source_sha256",
                      "frozen_case_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("Missing immutable score provenance")
        metrics = record.get("metrics") or {}
        duration = metrics.get("reference_speaker_seconds")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid reference speaker duration")
        for key_name in ("der", "missed_speaker_time", "false_alarm_time",
                         "confusion_time"):
            value = metrics.get(key_name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid or missing speaker error metric")
        cases[key] = record

    groups = {}
    for (meeting, model), record in cases.items():
        groups.setdefault(meeting, {})[model] = record
    selected = sorted(meeting for meeting, rows in groups.items()
                      if set(rows) == set(MODELS))
    for meeting in selected:
        rows = groups[meeting]
        for field in ("source_audio_sha256", "reference_source_sha256",
                      "frozen_case_sha256"):
            if len({row[field] for row in rows.values()}) != 1:
                raise ValueError(f"Nonpaired {field} on {meeting}")
        if len({row["metrics"]["reference_speaker_seconds"]
                for row in rows.values()}) != 1:
            raise ValueError(f"Different reference duration on {meeting}")
    # An unpaired result may be stored, but never counted as a comparison.
    full = len(selected) >= min_meetings
    totals = {}
    for model in MODELS:
        pieces = [groups[meeting][model]["metrics"] for meeting in selected]
        denominator = sum(p["reference_speaker_seconds"] for p in pieces)
        if not denominator:
            totals[model] = None
            continue
        metric_keys = ("der", "missed_speaker_time", "false_alarm_time",
                       "confusion_time")
        totals[model] = {
            key: round(
                sum(p[key] * p["reference_speaker_seconds"] for p in pieces)
                / denominator, 5
            )
            for key in metric_keys
        }
        totals[model]["reference_speaker_seconds"] = round(denominator, 2)
        totals[model]["meeting_count"] = len(selected)
    winner = min(MODELS, key=lambda name: (
        totals[name]["der"] if totals[name] is not None else math.inf
    )) if selected else None
    return {
        "schema": "meeting-phase1b-paired-comparison-v1",
        "paired_meetings": selected,
        "paired_meeting_count": len(selected),
        "min_meetings_required": min_meetings,
        "complete_heldout_cohort": full,
        "candidate_winner": winner if full else None,
        "weighted_metrics": totals,
        "acoustic_goal_met": bool(
            full and winner and totals[winner]["der"] < 0.20
        ),
        "may_auto_promote_to_live": False,
        "note": (
            "All rates are 100ms zero-collar and overlap-inclusive on "
            "the same AMI only_words data. The winner is only a research "
            "candidate, not evidence of real-time subtitle quality. "
            "Streaming Nemotron latency and offline candidate latency "
            "must not be compared as equivalent."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-meetings", type=int, default=10)
    options = parser.parse_args(argv)
    private = Path("benchmark/fixed195").resolve()
    report_paths = [p.resolve(strict=True) for p in options.reports]
    output = options.output.resolve()
    if (any(not path.is_relative_to(private) for path in report_paths)
            or not output.is_relative_to(private)
            or output.suffix != ".json" or output.exists()):
        parser.error("Read existing private inputs and write a NEW private JSON")
    result = compare_reports(
        [json.loads(p.read_text(encoding="utf-8")) for p in report_paths],
        min_meetings=options.min_meetings,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["acoustic_goal_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

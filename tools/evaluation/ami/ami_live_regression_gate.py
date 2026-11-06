"""Gate AMI live-replay changes against a frozen V6 reference.

This checks real-time *diarization* and annotated speech coverage. It is
NOT a WER/CER test and cannot establish the linguistic accuracy of newly
recovered ASR text. Unknown-speaker utterances remain usable content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _field(data: dict, *keys):
    value = data
    for key in keys:
        value = value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Missing numeric AMI field: " + ".".join(keys))
    return float(value)


def compare_live_audits(baseline: dict, candidate: dict) -> dict:
    """Reject mismatched audio/reference, missing provenance and regressions."""
    for name, data in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(data, dict) or not data.get("timeline_complete"):
            raise ValueError(f"{name} does not have a complete scored timeline")
        if not data.get("source_audio_sha256") or not data.get("reference_source_sha256"):
            raise ValueError(f"{name} missing frozen source/reference provenance")
        if data.get("collar_ms") != 0 or data.get("frame_step_ms") != 100:
            raise ValueError(f"{name} does not use the 100ms zero-collar AMI protocol")
        if not data.get("replay_sha256"):
            raise ValueError(f"{name} must originate from a concrete replay")
    fields = (
        "meeting", "source_audio_sha256", "reference_source_sha256",
        "reference_kind", "frame_step_ms", "collar_ms",
    )
    for key in fields:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Not paired AMI observations: {key}")
    if abs(_field(baseline, "metrics", "reference_speaker_seconds")
           - _field(candidate, "metrics", "reference_speaker_seconds")) > 0.001:
        raise ValueError("Reference-speaker scoring duration differs")

    probes = {
        "der": ("metrics", "der"),
        "missed_speaker_time": ("metrics", "missed_speaker_time"),
        "false_alarm_time": ("metrics", "false_alarm_time"),
        "confusion_time": ("metrics", "confusion_time"),
        "unidentified_speaker_seconds": (
            "missed_speaker_time_breakdown", "unidentified_transcript_s",
        ),
        "without_transcript_speaker_seconds": (
            "missed_speaker_time_breakdown", "without_transcript_s",
        ),
        "no_caption_clock_seconds": (
            "reference_speech_without_captions",
            "reference_speech_without_captions_seconds",
        ),
    }
    values = {}
    for name, field in probes.items():
        before = _field(baseline, *field)
        after = _field(candidate, *field)
        values[name] = {
            "baseline": round(before, 5),
            "candidate": round(after, 5),
            "delta": round(after - before, 5),
        }

    # Non-regression tolerance is only a guard against timing noise; it
    # never upgrades an ASR text hypothesis to verified linguistic truth.
    gates = {
        "der_not_worse_by_over_0p5pp": values["der"]["delta"] <= 0.005,
        "missed_time_not_worse_by_over_0p5pp":
            values["missed_speaker_time"]["delta"] <= 0.005,
        "false_alarm_not_worse_by_over_0p5pp":
            values["false_alarm_time"]["delta"] <= 0.005,
        "confusion_not_worse_by_over_0p5pp":
            values["confusion_time"]["delta"] <= 0.005,
        "no_caption_clock_time_not_worse_by_over_0p5s":
            values["no_caption_clock_seconds"]["delta"] <= 0.5,
        "without_transcript_speaker_time_not_worse_by_over_0p5s":
            values["without_transcript_speaker_seconds"]["delta"] <= 0.5,
    }
    # Speaker-count and overlap guardrails apply when both paired audits
    # report the corresponding reference-backed measurements.
    for metric, comparison in (
        ("speaker_count_absolute_error", "not_increase"),
        ("overlap_all_speakers_recall", "not_decrease"),
    ):
        before = baseline.get("metrics", {}).get(metric)
        after = candidate.get("metrics", {}).get(metric)
        if (isinstance(before, (int, float)) and not isinstance(before, bool)
                and isinstance(after, (int, float)) and not isinstance(after, bool)):
            gates[metric + "_not_worse"] = (
                after <= before if comparison == "not_increase"
                else after >= before
            )

    # A permissive non-regression pass is not a quality improvement. Require
    # independently meaningful reductions before describing a candidate as
    # better on this frozen diarization/coverage benchmark.
    measurable_improvement = (
        all(gates.values())
        and values["der"]["delta"] <= -0.005
        and values["missed_speaker_time"]["delta"] < 0
        and values["no_caption_clock_seconds"]["delta"] <= -0.5
        and values["without_transcript_speaker_seconds"]["delta"] <= 0
    )
    return {
        "schema": "meeting-ami-live-regression-gate-v1",
        "meeting": baseline["meeting"],
        "non_regression_pass": all(gates.values()),
        "diarization_coverage_improvement_pass": measurable_improvement,
        "improvement_rule": (
            "Requires DER reduction >=0.5 percentage points, lower missed "
            "speaker time, >=0.5s less reference speech without captions, "
            "no increase in transcript-absent speaker time, and every "
            "non-regression gate. Not a WER/CER or overlap-quality claim."
        ),
        "gates": gates,
        "measurements": values,
        "caveat": (
            "A non-regression result is NOT evidence of recovered true words. "
            "DER counts unidentified speaker-time as missed even when text "
            "exists. AMI RTTM only_words is not a reliable reference "
            "transcription for WER/CER. Separately review real text, "
            "duplicate/hallucinated captions, inference latency, and "
            "audio transport before claiming a stage-one improvement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--require-improvement", action="store_true",
        help="Exit nonzero unless measured DER and subtitle coverage improve",
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_live_audits(baseline, candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    accepted = (
        result["diarization_coverage_improvement_pass"]
        if args.require_improvement else result["non_regression_pass"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

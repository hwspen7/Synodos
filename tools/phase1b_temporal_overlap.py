"""Time-aligned overlap attribution for five existing AMI Pyannote failures.

Runs actual GPU Pyannote on frozen audio. Derives all three DER components
separately on silence, single-speaker, and overlapped reference frames.
Uses the same global speaker assignment and 100-ms grid as the frozen
baseline scorer. Never edits production speaker labels or baseline reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from speech.phase1b_diarization_models import MODELS, _validate_model_path, run_model
from tools.ami_shadow10_prepare import DURATION_S, SELECTED
from tools.phase09_speaker_benchmark import _intervals, evaluate
from tools.phase1b_diarization_benchmark import pinned_case
from tools.phase1b_pyannote_ami10 import validate_report
from tools.speaker_stage1_compare import _reference_turns

STEP_MS = 100
TARGETS = ("EN2002d", "EN2002a", "EN2002b", "ES2004d", "TS3003b")
METRICS = ("der", "missed_speaker_time", "false_alarm_time", "confusion_time")
SCHEMA = "meeting-phase1b-temporal-overlap-audit-v1"


def _counter() -> dict:
    return {"frames": 0, "reference_speaker_frames": 0,
            "missed": 0, "false_alarm": 0, "confusion": 0,
            "all_correct_overlap_frames": 0,
            "overlap_underactive_frames": 0}


def _rates(counter: dict) -> dict:
    denom = counter["reference_speaker_frames"]
    return {
        "frame_count": counter["frames"],
        "reference_speaker_seconds": round(denom * STEP_MS / 1000, 2),
        "missed_speaker_seconds": round(counter["missed"] * STEP_MS / 1000, 2),
        "false_alarm_speaker_seconds": round(counter["false_alarm"] * STEP_MS / 1000, 2),
        "confusion_speaker_seconds": round(counter["confusion"] * STEP_MS / 1000, 2),
        "der": round((
            counter["missed"] + counter["false_alarm"] + counter["confusion"]
        ) / denom, 5) if denom else None,
        "missed_rate": round(counter["missed"] / denom, 5) if denom else None,
        "false_alarm_rate": round(counter["false_alarm"] / denom, 5) if denom else None,
        "confusion_rate": round(counter["confusion"] / denom, 5) if denom else None,
        "all_correct_overlap_fraction": round(
            counter["all_correct_overlap_frames"] / counter["frames"], 5
        ) if counter["frames"] else None,
        "overlap_underactive_fraction": round(
            counter["overlap_underactive_frames"] / counter["frames"], 5
        ) if counter["frames"] else None,
    }


def _add(bucket: dict, reference: set, predicted: set, mapping: dict) -> None:
    r, h = len(reference), len(predicted)
    mapped = {mapping.get(p) for p in predicted}
    matched = len(reference.intersection(mapped))
    bucket["frames"] += 1
    bucket["reference_speaker_frames"] += r
    bucket["missed"] += max(0, r - h)
    bucket["false_alarm"] += max(0, h - r)
    bucket["confusion"] += max(0, min(r, h) - matched)
    if r >= 2:
        bucket["overlap_underactive_frames"] += h < r
        bucket["all_correct_overlap_frames"] += reference.issubset(mapped)


def stratify(reference: list[dict], hypothesis: list[dict], *,
             duration_ms: int = DURATION_S * 1000,
             bin_ms: int = 30000) -> dict:
    """Stratify the fixed scorer without remapping independently per bucket."""
    if duration_ms <= 0 or duration_ms % STEP_MS or bin_ms % STEP_MS:
        raise ValueError("Invalid frozen reference grid or time bin")
    ref = _intervals(reference, "speaker")
    pred = _intervals(hypothesis, "speaker_id")
    if not ref or any(b > duration_ms for _, b, _ in ref + pred):
        raise ValueError("Missing or out-of-bounds reference/hypothesis")
    score = evaluate(reference, hypothesis, step_ms=STEP_MS)
    mapping = score["speaker_mapping"]
    buckets = {key: _counter() for key in ("silence", "single", "overlap")}
    by_bin: dict[int, dict] = {}
    for time_ms in range(0, duration_ms, STEP_MS):
        reference_now = {name for a, b, name in ref if a <= time_ms < b}
        predicted_now = {name for a, b, name in pred if a <= time_ms < b}
        name = ("silence" if len(reference_now) == 0 else
                "single" if len(reference_now) == 1 else "overlap")
        _add(buckets[name], reference_now, predicted_now, mapping)
        bin_no = time_ms // bin_ms
        _add(by_bin.setdefault(bin_no, _counter()),
             reference_now, predicted_now, mapping)
    totals = {key: sum(b[key] for b in buckets.values())
              for key in ("reference_speaker_frames", "missed",
                          "false_alarm", "confusion")}
    if not totals["reference_speaker_frames"]:
        raise ValueError("Empty reference speaker time")
    computed = {
        "der": (totals["missed"] + totals["false_alarm"]
                + totals["confusion"]) / totals["reference_speaker_frames"],
        "missed_speaker_time": totals["missed"] / totals["reference_speaker_frames"],
        "false_alarm_time": totals["false_alarm"] / totals["reference_speaker_frames"],
        "confusion_time": totals["confusion"] / totals["reference_speaker_frames"],
    }
    if abs(score["reference_speaker_seconds"] -
           totals["reference_speaker_frames"] * STEP_MS / 1000) > .001:
        raise ValueError("Temporal scorer denominator differs from frozen scorer")
    if any(abs(computed[k] - score[k]) > 0.000011 for k in METRICS):
        raise ValueError("Temporal error components do not conserve original DER")
    ref_overlap = buckets["overlap"]["reference_speaker_frames"]
    miss_total = totals["missed"]
    windows = sorted([
        {
            "start_seconds": index * bin_ms / 1000,
            "end_seconds": min(duration_ms, (index + 1) * bin_ms) / 1000,
            "reference_speaker_seconds": round(
                c["reference_speaker_frames"] * STEP_MS / 1000, 2),
            "missed_speaker_seconds": round(c["missed"] * STEP_MS / 1000, 2),
            "confusion_speaker_seconds": round(c["confusion"] * STEP_MS / 1000, 2),
            "false_alarm_speaker_seconds": round(c["false_alarm"] * STEP_MS / 1000, 2),
            "der": _rates(c)["der"],
        }
        for index, c in by_bin.items()
    ], key=lambda x: x["start_seconds"])
    return {
        "schema": SCHEMA,
        "scoring": {"step_ms": STEP_MS, "collar_ms": 0,
                    "overlap_included": True, "speaker_mapping_scope": "whole_recording"},
        "overall": {k: round(v, 5) for k, v in computed.items()},
        "reference_speaker_seconds": score["reference_speaker_seconds"],
        "reference_speaker_count": len({r[2] for r in ref}),
        "predicted_speaker_count": len({h[2] for h in pred}),
        "conditions": {k: _rates(v) for k, v in buckets.items()},
        "overlap_fraction_of_reference_speaker_time": round(
            ref_overlap / totals["reference_speaker_frames"], 5),
        "share_of_missed_speaker_time_during_overlap": round(
            buckets["overlap"]["missed"] / miss_total, 5
        ) if miss_total else None,
        "worst_30_second_windows": sorted(
            [w for w in windows if w["der"] is not None],
            key=lambda w: (-w["der"], w["start_seconds"])
        )[:3],
        "not_oracle": (
            "No reference speaker count, speaker labels, or timing is supplied "
            "to the model. Gold references are used only after inference."
        ),
    }


def diagnose(root: Path, folder: Path, *, meetings: tuple[str, ...],
             allow_download: bool = False) -> None:
    private = Path("benchmark/fixed195").resolve()
    root = root.resolve(strict=True)
    folder = folder.resolve(strict=True)
    if not root.is_relative_to(private) or not folder.is_relative_to(root / "phase1b"):
        raise ValueError("Pinned AMI source and report folder must stay in private Fixed195")
    if os.name == "nt" and root.drive.upper() != "E:":
        raise ValueError("GPU evaluation must use the E-drive worktree")
    cohort_path = folder / "pyannote_ami10_summary.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if (cohort.get("schema") != "meeting-phase1b-pyannote-ami10-cohort-v1"
            or cohort.get("meeting_count") != len(SELECTED)):
        raise ValueError("Full frozen Pyannote cohort summary required")
    if not meetings or len(meetings) != len(set(meetings)) or any(
        m not in dict(SELECTED) for m in meetings
    ):
        raise ValueError("Use distinct frozen AMI meeting IDs only")
    verified = {}
    for meeting in meetings:
        frozen = json.loads(
            (folder / f"pyannote-{meeting}.json").read_text(encoding="utf-8")
        )
        validate_report(frozen, root, meeting)
        wav, case, case_hash = pinned_case(root, meeting)
        verified[meeting] = (frozen, wav, case, case_hash)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. No CPU fallback for temporal audit")
    from pyannote.audio import Pipeline
    model = _validate_model_path(MODELS["pyannote"], allow_download)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading Community-1 once. Using original model settings, no oracle speaker count.")
    pipeline = Pipeline.from_pretrained(
        model, **({"token": token} if token else {})
    )
    if pipeline is None:
        raise RuntimeError("Pyannote weights unavailable")
    pipeline.to(torch.device("cuda"))
    destination = folder / (
        "overlap_attribution-" +
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") +
        f"-{os.getpid()}"
    )
    destination.mkdir(exist_ok=False)
    drifted = []
    for meeting in meetings:
        frozen, wav, case, case_hash = verified[meeting]
        target = destination / f"{meeting}.json"
        print(f"Analyzing {meeting}...", flush=True)
        turns, model_diag = run_model(
            "pyannote", wav, duration_ms=DURATION_S * 1000,
            device="cuda", pipeline=pipeline
        )
        reference = _reference_turns(case["reference_turns"], DURATION_S * 1000)
        result = stratify(reference, turns)
        drift = {
            k: round(result["overall"][k] - frozen["metrics"][k], 5)
            for k in METRICS
        }
        result.update({
            "meeting": meeting,
            "source_audio_sha256": case["sha256"],
            "reference_source_sha256": case["reference_source_sha256"],
            "frozen_case_sha256": case_hash,
            "model_id": MODELS["pyannote"],
            "recomputed_v_frozen_metric_delta": drift,
            "inference_seconds": model_diag["wall_seconds_including_load"],
            "compatible_with_original_cohort": all(
                abs(v) <= .002 for v in drift.values()
            ),
            "interpretation_limit": (
                "These are frame-level observational strata, not causal "
                "proof that overlap caused missing speech. Re-running a "
                "remote model without an immutable weight revision can drift."
            ),
        })
        with target.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        c = result["conditions"]
        fmt = lambda n: f"{n * 100:.2f}%" if n is not None else "N/A"
        status = "MATCH" if result["compatible_with_original_cohort"] else "DRIFT"
        if status == "DRIFT":
            drifted.append(meeting)
        print(f"{meeting}: baseline {fmt(frozen['metrics']['der'])} | "
              f"rerun {fmt(result['overall']['der'])} ({status}) | "
              f"miss-solo {c['single']['missed_speaker_seconds']:.1f}s | "
              f"miss-overlap {c['overlap']['missed_speaker_seconds']:.1f}s | "
              f"overlap full recall {fmt(c['overlap']['all_correct_overlap_fraction'])}")
    print("Private time-aligned reports:", destination)
    print("No production configuration changed.")
    if drifted:
        raise RuntimeError(
            "Recomputed model results differ from saved baseline on "
            + ", ".join(drifted)
            + ". Reports preserved but cross-condition conclusions are NOT accepted."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=Path("benchmark/fixed195/ami_shadow10"))
    parser.add_argument("--cohort-summary", type=Path, required=True)
    parser.add_argument("--meetings", nargs="+", default=list(TARGETS))
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args(argv)
    if args.cohort_summary.name != "pyannote_ami10_summary.json":
        parser.error("Use the actual ten-meeting cohort summary")
    diagnose(args.root, args.cohort_summary.parent,
             meetings=tuple(args.meetings),
             allow_download=args.allow_model_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

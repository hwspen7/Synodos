"""Run one genuine Pyannote Community-1 model load over the ten frozen AMI cases.

All files remain in the private Fixed195 worktree. Never promote a model
automatically, and never mistake one passing meeting for a held-out cohort.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.asr.diarization_benchmark import (
    SCHEMA, benchmark, pinned_case,
)

SCORE_PROTOCOL = {
    "frame_step_ms": 100,
    "collar_ms": 0,
    "reference_protocol": "only_words",
    "overlap_included": True,
}
METRICS = ("der", "missed_speaker_time", "false_alarm_time", "confusion_time")


def validate_report(report: dict, root: Path, meeting: str) -> dict:
    """Verify previously computed evidence against the *current* pinned case."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"{meeting}: invalid Phase 1B report schema")
    if report.get("model") != "pyannote" or report.get("meeting") != meeting:
        raise ValueError(f"{meeting}: not a Pyannote report for this meeting")
    _, case, case_hash = pinned_case(root, meeting)
    expected = {
        "source_audio_sha256": case["sha256"],
        "reference_source_sha256": case["reference_source_sha256"],
        "frozen_case_sha256": case_hash,
        **SCORE_PROTOCOL,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{meeting}: report has incorrect {key}")
    diag = report.get("diagnostics")
    if not isinstance(diag, dict) or (
        diag.get("model_id") != "pyannote/speaker-diarization-community-1"
    ):
        raise ValueError(f"{meeting}: unpinned Pyannote model provenance")
    if diag.get("device") != "cuda":
        raise ValueError(f"{meeting}: report is not from CUDA evaluation")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{meeting}: absent model metrics")
    for key in (*METRICS, "reference_speaker_seconds"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{meeting}: invalid metric {key}")
        if value < 0 or (key == "reference_speaker_seconds" and value == 0):
            raise ValueError(f"{meeting}: invalid metric {key}")
    if abs(
        metrics["der"] - sum(metrics[key] for key in METRICS[1:])
    ) > 0.002:
        raise ValueError(f"{meeting}: DER component totals disagree")
    if report.get("accepted_acoustic_candidate") is not (
        metrics["der"] < 0.20
    ):
        raise ValueError(f"{meeting}: acceptance flag differs from DER")
    return report


def summarize(reports: list[dict], root: Path) -> dict:
    """Compute speaker-time weighted metrics, never averaging ten DER values."""
    expected = [meeting for meeting, _ in SELECTED]
    found = [record.get("meeting") for record in reports]
    if len(reports) != len(expected) or len(set(found)) != len(expected):
        raise ValueError("Exactly ten unique AMI meeting reports are required")
    if set(found) != set(expected):
        raise ValueError("AMI meeting cohort is incomplete or altered")
    validated = {m: validate_report(r, root, m) for r in reports
                 for m in [r["meeting"]]}
    ordered = [validated[m] for m in expected]
    total = sum(r["metrics"]["reference_speaker_seconds"] for r in ordered)
    weighted = {
        key: round(sum(
            r["metrics"][key] * r["metrics"]["reference_speaker_seconds"]
            for r in ordered
        ) / total, 5)
        for key in METRICS
    }
    rows = [{
        "meeting": r["meeting"],
        "der": r["metrics"]["der"],
        "missed_speaker_time": r["metrics"]["missed_speaker_time"],
        "false_alarm_time": r["metrics"]["false_alarm_time"],
        "confusion_time": r["metrics"]["confusion_time"],
        "reference_speaker_seconds": r["metrics"]["reference_speaker_seconds"],
        "single_meeting_der_below_20_percent": r["metrics"]["der"] < 0.20,
        "reported_wall_seconds": r["diagnostics"].get(
            "wall_seconds_including_load"
        ),
        "timing_scope": (
            "preloaded_model_inference" if r["diagnostics"].get(
                "model_preloaded_for_cohort"
            ) else "standalone_model_load_plus_inference"
        ),
    } for r in ordered]
    return {
        "schema": "meeting-phase1b-pyannote-ami10-cohort-v1",
        "model": "pyannote/speaker-diarization-community-1",
        "reference_protocol": "only_words",
        "frame_step_ms": 100,
        "collar_ms": 0,
        "overlap_included": True,
        "meeting_count": len(rows),
        "reference_speaker_seconds": round(total, 2),
        "weighted_metrics": weighted,
        "meetings_below_20_percent_der": sum(
            row["single_meeting_der_below_20_percent"] for row in rows
        ),
        "overall_acoustic_goal_met": weighted["der"] < 0.20,
        "all_meetings_below_20_percent_der": all(
            row["single_meeting_der_below_20_percent"] for row in rows
        ),
        "per_meeting": rows,
        "automatic_model_promotion": False,
        "remote_model_weights_exact_revision_pinned": False,
        "not_validated": [
            "streaming latency", "word error rate",
            "cross-meeting voiceprint identification",
            "commercial model license", "live default model quality",
        ],
    }


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _write_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def execute(root: Path, run_dir: Path, *, reuse_report: Path | None = None,
            resume: bool = False, allow_download: bool = False) -> dict:
    root = root.resolve(strict=True)
    private = Path("benchmark/fixed195").resolve()
    if not _inside(root, private):
        raise ValueError("Only private Fixed195 AMI inputs are supported")
    if os.name == "nt" and root.drive.upper() != "E:":
        raise ValueError("Run from the E: benchmark worktree")
    missing = []
    for meeting, _ in SELECTED:
        try:
            pinned_case(root, meeting)  # complete preflight before GPU work
        except FileNotFoundError:
            missing.append(meeting)
    if missing:
        raise FileNotFoundError(
            "Missing frozen AMI recordings or labels: "
            + ", ".join(missing)
            + ". Prepare with: python -m tools.evaluation.ami.ami_shadow10_prepare"
        )

    approved_parent = (root / "phase1b").resolve()
    run_dir = run_dir.resolve()
    if run_dir.parent != approved_parent or (
        not run_dir.name.startswith("pyannote10-")
    ):
        raise ValueError("Run directory must be a named pyannote10-* folder under phase1b")
    if resume:
        if not run_dir.is_dir():
            raise ValueError("Cannot resume a missing run directory")
    else:
        approved_parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)

    if reuse_report:
        reuse_report = reuse_report.resolve(strict=True)
        if not _inside(reuse_report, private):
            raise ValueError("Reuse input must remain under private Fixed195")
        source = json.loads(reuse_report.read_text(encoding="utf-8"))
        validate_report(source, root, "TS3003b")
        dest = run_dir / "pyannote-TS3003b.json"
        if not dest.exists():
            _write_exclusive(dest, source)
        print("Reused: TS3003b (frozen hashes and scoring protocol checked)", flush=True)

    completed = {}
    for meeting, _ in SELECTED:
        existing = run_dir / f"pyannote-{meeting}.json"
        if existing.is_file():
            completed[meeting] = validate_report(
                json.loads(existing.read_text(encoding="utf-8")), root, meeting
            )
    remaining = [m for m, _ in SELECTED if m not in completed]
    print(f"AMI cohort: {len(completed)} existing / {len(remaining)} GPU evaluations", flush=True)

    pipeline = None
    if remaining:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable; no CPU fallback")
        from pyannote.audio import Pipeline
        from speech.diarization.model_candidates import MODELS, _validate_model_path
        model_id = _validate_model_path(MODELS["pyannote"], allow_download)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print("Loading Pyannote Community-1 once for the cohort...", flush=True)
        pipeline = Pipeline.from_pretrained(model_id, **(
            {"token": token} if token else {}
        ))
        if pipeline is None:
            raise RuntimeError("Failed to load Community-1; verify license and HF_TOKEN")
        pipeline.to(torch.device("cuda"))

    for index, (meeting, _) in enumerate(SELECTED, 1):
        output = run_dir / f"pyannote-{meeting}.json"
        if meeting not in completed:
            print(f"[{index}/10] GPU inference {meeting}...", flush=True)
            # benchmark() checks WAV checksum and human reference again.
            report = benchmark(
                root, meeting, "pyannote", device="cuda",
                injected_pipeline=pipeline
            )
            report["diagnostics"]["model_preloaded_for_cohort"] = True
            validate_report(report, root, meeting)
            _write_exclusive(output, report)
            completed[meeting] = report
        value = completed[meeting]["metrics"]["der"]
        print(f"[{index}/10] {meeting}: DER {value * 100:.2f}% "
              f"({'pass' if value < .20 else 'above target'})", flush=True)

    summary = summarize(list(completed.values()), root)
    aggregate = run_dir / "pyannote_ami10_summary.json"
    if aggregate.exists():
        raise FileExistsError("Cohort summary already exists; do not overwrite")
    _write_exclusive(aggregate, summary)
    print("\n=== AMI 10-MEETING WEIGHTED RESULTS ===")
    for key in METRICS:
        print(f"{key}: {summary['weighted_metrics'][key] * 100:.2f}%")
    print(f"Meetings below 20%: {summary['meetings_below_20_percent_der']}/10")
    print("Weighted DER gate:", "PASS" if summary["overall_acoustic_goal_met"] else "FAIL")
    print("Real-time rollout: NOT APPROVED; Phase 2 FAR/FRR: NOT MEASURED")
    print("Private summary:", aggregate)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195/ami_shadow10"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-report", type=Path)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    run_dir = args.run_dir or (
        args.root / "phase1b" /
        ("pyannote10-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
         + f"-{os.getpid()}")
    )
    summary = execute(
        args.root, run_dir, reuse_report=args.reuse_report,
        resume=args.resume, allow_download=args.allow_model_download,
    )
    return 0 if summary["overall_acoustic_goal_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

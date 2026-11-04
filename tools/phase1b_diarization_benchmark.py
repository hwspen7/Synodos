"""Phase 1B frozen AMI single-model comparison, no model promotion.

Examples:
 python -m tools.phase1b_diarization_benchmark --root benchmark/fixed195/ami_shadow10 --meeting TS3003b --model pyannote --device cuda --allow-model-download --output benchmark/fixed195/ami_shadow10/phase1b/pyannote_TS3003b.json
 python -m tools.phase1b_diarization_benchmark --root benchmark/fixed195/ami_shadow10 --meeting TS3003b --model nemotron3 --replay benchmark/fixed195/ami_shadow10/replay_TS3003b_nemotron_....json --output benchmark/fixed195/ami_shadow10/phase1b/nemotron3_TS3003b.json

These are independent model experiments, not the production caption DER.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from speech.phase1b_diarization_models import MODELS, run_model
from tools.ami_shadow10_prepare import (
    DURATION_S, SELECTED, checksum, validate_reference,
)
from tools.phase09_speaker_benchmark import evaluate
from tools.speaker_stage1_compare import _reference_turns


SCHEMA = "meeting-phase1b-frozen-diarization-v1"


def pinned_case(root: Path, meeting: str) -> tuple[Path, dict, str]:
    offsets = dict(SELECTED)
    if meeting not in offsets:
        raise ValueError("Meeting is not in the ten frozen AMI meetings")
    meta = root / "cases" / f"{meeting}.json"
    raw = meta.read_bytes()
    case = json.loads(raw)
    offset = offsets[meeting]
    audio_name = f"audio/ami-{meeting}-{offset}-{offset + DURATION_S}.wav"
    if (case.get("source_meeting") != meeting
            or case.get("start_seconds") != offset
            or case.get("duration_seconds") != DURATION_S
            or case.get("audio") != audio_name
            or not isinstance(case.get("reference_source_sha256"), str)
            or len(case["reference_source_sha256"]) != 64):
        raise ValueError("Frozen AMI case metadata is inconsistent")
    wav = (root / audio_name).resolve()
    if not wav.is_relative_to(root.resolve()) or not wav.is_file():
        raise ValueError("Frozen WAV is not within the benchmark root")
    if checksum(wav) != case.get("sha256"):
        raise ValueError("Frozen AMI audio SHA256 mismatch")
    validate_reference(case["reference_turns"])
    return wav, case, sha256(raw).hexdigest()


def report_from_turns(
    case: dict, case_sha: str, model: str, turns: list[dict],
    diagnostic: dict, *, max_der: float = 0.20,
) -> dict:
    reference = _reference_turns(case["reference_turns"], DURATION_S * 1000)
    result = evaluate(reference, turns, step_ms=100)
    result.pop("speaker_mapping", None)
    score = result["der"]
    return {
        "schema": SCHEMA,
        "meeting": case["source_meeting"],
        "model": model,
        "source_audio_sha256": case["sha256"],
        "reference_source_sha256": case["reference_source_sha256"],
        "frozen_case_sha256": case_sha,
        "reference_protocol": "only_words",
        "frame_step_ms": 100,
        "collar_ms": 0,
        "overlap_included": True,
        "metrics": result,
        "diagnostics": diagnostic,
        "accepted_acoustic_candidate": bool(
            score is not None and score < max_der
        ),
        "max_der_exclusive": max_der,
        "limitations": (
            "One frozen AMI recording is not a held-out multi-meeting "
            "validation. Independent acoustic DER is not subtitle CER, "
            "speaker identity verification, or proof that overlap words "
            "were transcribed. Offline and streaming latency are not "
            "directly comparable."
        ),
    }


def benchmark(
    root: Path, meeting: str, model: str, *, device: str = "cpu",
    replay: Path | None = None, allow_model_download: bool = False,
    model_id: str | None = None, injected_pipeline: Any = None,
) -> dict:
    root = root.resolve(strict=True)
    wav, case, case_sha = pinned_case(root, meeting)
    if model == "nemotron3":
        if replay is None:
            raise ValueError("Nemotron requires a completed independent replay")
        replay = replay.resolve(strict=True)
        if not replay.is_relative_to(root):
            raise ValueError("Nemotron replay must be in the frozen benchmark")
        from tools.ami_shadow10_score import score_case
        from tools.ami_nemotron_live_score import _validated_independent
        score_case(root, meeting, replay)  # checks reference and source pins
        data = json.loads(replay.read_text(encoding="utf-8"))
        if data.get("server_ready", {}).get("independent_diarization") != "nemotron3":
            raise ValueError("Not a Nemotron-enabled replay")
        turns = _validated_independent(data)
        diagnostics = {
            "model": "nemotron3",
            "inference_mode": "streaming_saved_replay",
            "model_id": "nvidia/Nemotron-3-Diarization",
            "replay_sha256": checksum(replay),
            "device": "recorded_in_replay",
            "wall_seconds_including_load": None,
        }
    elif model in MODELS:
        if replay is not None:
            raise ValueError("Offline candidate cannot take a Nemotron replay")
        turns, diagnostics = run_model(
            model, wav, duration_ms=DURATION_S * 1000,
            device=device, model_id=model_id,
            allow_download=allow_model_download,
            pipeline=injected_pipeline,
        )
    else:
        raise ValueError(f"Unknown Phase 1B model: {model}")
    return report_from_turns(case, case_sha, model, turns, diagnostics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--meeting", required=True, choices=[k for k, _ in SELECTED])
    parser.add_argument("--model", choices=["nemotron3", *MODELS], required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    private = Path("benchmark/fixed195").resolve()
    if not root.is_relative_to(private):
        parser.error("Frozen benchmark root must be in private benchmark/fixed195")
    output = args.output.resolve()
    if (not output.is_relative_to(private) or output.suffix != ".json"
            or output.exists()):
        parser.error("Output must be a NEW private benchmark/fixed195 JSON")
    report = benchmark(
        root, args.meeting, args.model, device=args.device,
        replay=args.replay, model_id=args.model_id,
        allow_model_download=args.allow_model_download,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "meeting": report["meeting"],
        "model": report["model"],
        "metrics": report["metrics"],
        "accepted_acoustic_candidate": report["accepted_acoustic_candidate"],
        "output": str(output),
    }, indent=2))
    return 0 if report["accepted_acoustic_candidate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

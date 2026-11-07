"""Compare NVIDIA Nemotron-3 offline speaker turns against one AMI case.

Splits a four-minute excerpt into two independent 120-second inference
windows, so no untested cross-window speaker identity is fabricated.
Both the CAM++ replay and Nemotron are scored separately on each window
against the SAME frozen AMI human-reference RTTM. A pooled DER here is
window-wise, NOT cross-window speaker tracking or a live latency result.

Run from .venv-speaker-nemotron; it is never loaded in the live backend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.validation.dialogue_speaker_benchmark import evaluate
from tools.evaluation.speakers.speaker_stage1_compare import baseline_turns, _reference_turns


WINDOW_SECONDS = 120


def read_windows(source: Path) -> list[np.ndarray]:
    with wave.open(str(source), "rb") as audio:
        fmt = (
            audio.getframerate(), audio.getnchannels(),
            audio.getsampwidth(), audio.getcomptype(),
        )
        if fmt != (16000, 1, 2, "NONE") or audio.getnframes() != 240 * 16000:
            raise ValueError("Expected frozen 240-second mono PCM16 AMI WAV")
        frames = audio.readframes(240 * 16000)
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return [np.ascontiguousarray(pcm[i * 16000 * WINDOW_SECONDS:
                                     (i + 1) * 16000 * WINDOW_SECONDS])
            for i in range(2)]


def clip_turns(turns: list[dict], start_ms: int, duration_ms: int,
               identity_key: str) -> list[dict]:
    """Crop original timestamps to one window without losing overlaps."""
    end_ms = start_ms + duration_ms
    result = []
    for item in turns:
        a = max(start_ms, int(item["start_ms"]))
        b = min(end_ms, int(item["end_ms"]))
        if b > a:
            result.append({
                "start_ms": a - start_ms, "end_ms": b - start_ms,
                identity_key: item[identity_key],
            })
    return result


def compare_windows(
    reference: list[dict], baseline: list[dict],
    audio: list[np.ndarray], model,
) -> dict:
    if len(audio) != 2 or any(len(x) != 120 * 16000 for x in audio):
        raise ValueError("Must compare exactly two 120-second AMI audio windows")
    windows = []
    for index, samples in enumerate(audio):
        start_ms = index * WINDOW_SECONDS * 1000
        reference_local = clip_turns(
            reference, start_ms, WINDOW_SECONDS * 1000, "speaker",
        )
        baseline_local = clip_turns(
            baseline, start_ms, WINDOW_SECONDS * 1000, "speaker_id",
        )
        if not reference_local:
            raise ValueError("Human reference missing within AMI window")
        prediction_local = model.diarize(samples)
        nemotron = evaluate(reference_local, prediction_local)
        existing = evaluate(reference_local, baseline_local)
        windows.append({
            "window_index": index,
            "start_ms": start_ms,
            "end_ms": start_ms + WINDOW_SECONDS * 1000,
            "reference_speakers": len({
                row["speaker"] for row in reference_local
            }),
            "baseline": existing,
            "nemotron3": nemotron,
        })

    def weighted_error(key: str, category: str) -> float | None:
        total = sum(
            w["reference_speakers"] >= 1
            and w[category]["reference_speaker_seconds"]
            for w in windows
        )
        if not total:
            return None
        return round(sum(
            (w[category][key] or 0) * w[category]["reference_speaker_seconds"]
            for w in windows
        ) / total, 5)

    return {
        "window_seconds": WINDOW_SECONDS,
        "window_count": 2,
        "model_id": getattr(model, "model_id", "test-backend"),
        "windows": windows,
        "independent_window_weighted_der": {
            "camplus_baseline": weighted_error("der", "baseline"),
            "nemotron3": weighted_error("der", "nemotron3"),
        },
        "cross_window_identity_evaluable": False,
        "not_live_equivalent": True,
        "caveat": (
            "Only within-window anonymous speaker identities are compared; "
            "speaker slots restart on every 120-second chunk. Do not treat "
            "these numbers as 240-second cross-session DER, model accuracy "
            "for known identities, or production suitability."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark/fixed195/ami_shadow10"),
    )
    parser.add_argument("--meeting", choices=[m for m, _ in SELECTED], required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    private = Path("benchmark/fixed195").resolve()
    if (not root.is_relative_to(private)
            or not output.is_relative_to(root)
            or output.suffix.lower() != ".json"
            or output.exists()):
        parser.error("Output must be a new private Fixed195 JSON file")
    # Auditing validates source WAV hash and the pinned RTTM against this replay.
    score_case(root, args.meeting, args.replay)
    case = json.loads(
        (root / "cases" / (args.meeting + ".json")).read_text(encoding="utf-8")
    )
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    audio_path = (root / case["audio"]).resolve()
    baseline = baseline_turns(replay, duration_ms=240000)
    reference = _reference_turns(case["reference_turns"], 240000)
    from speech.diarization.speaker_backends import create_diarization_backend
    model = create_diarization_backend("nemotron3", device=args.device)
    result = compare_windows(reference, baseline, read_windows(audio_path), model)
    result["schema"] = "meeting-ami-nemotron-independent-window-compare-v1"
    result["meeting"] = args.meeting
    result["audio_sha256"] = case["sha256"]
    result["reference_sha256"] = case["reference_source_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "meeting": args.meeting,
        "weighted_der": result["independent_window_weighted_der"],
        "windows": [
            {"start_ms": x["start_ms"], "end_ms": x["end_ms"],
             "reference_speakers": x["reference_speakers"],
             "baseline_der": x["baseline"]["der"],
             "nemotron_der": x["nemotron3"]["der"]}
            for x in result["windows"]
        ],
        "output": str(output),
        "not_live_equivalent": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

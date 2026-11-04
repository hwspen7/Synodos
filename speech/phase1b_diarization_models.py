"""Optional offline diarization models for frozen Phase 1B comparisons.

Do not load weights at import time. Model downloads require an explicit
operator flag; this module never touches the live caption pipeline.
DiariZen pre-trained weights are CC BY-NC 4.0 and must not be distributed
as part of this project.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any


MODELS = {
    "pyannote": "pyannote/speaker-diarization-community-1",
    "diarizen": "BUT-FIT/diarizen-wavlm-large-s80-md",
}


class ModelSetupError(RuntimeError):
    """The optional model environment or weights are unavailable."""


def _normalize_turns(annotation: Any, duration_ms: int) -> list[dict]:
    """Convert anonymous outputs to local numeric IDs, retaining overlap."""
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    if hasattr(annotation, "itertracks"):
        iterator = ((span, label)
                    for span, _, label in annotation.itertracks(yield_label=True))
    else:
        iterator = iter(annotation)
    entries = []
    for span, label in iterator:
        a, b = float(span.start), float(span.end)
        if (not math.isfinite(a) or not math.isfinite(b)
                or a < 0 or b <= a
                or b * 1000 > duration_ms + 50):
            raise ValueError("Model returned invalid or out-of-range speaker interval")
        label = str(label)
        if not label or len(label) > 128:
            raise ValueError("Model returned an invalid speaker label")
        x, y = round(a * 1000), min(duration_ms, round(b * 1000))
        if y > x:
            entries.append((x, y, label))
        if len(entries) > 20000:
            raise ValueError("Diarization output exceeds safe turn limit")
    entries.sort(key=lambda row: (row[0], row[1], row[2]))
    labels: dict[str, int] = {}
    turns = []
    for a, b, label in entries:
        labels.setdefault(label, len(labels) + 1)
        turns.append({"start_ms": a, "end_ms": b,
                      "speaker_id": labels[label]})
    return turns


def _validate_model_path(model_id: str, allow_download: bool) -> str:
    if not model_id or len(model_id) > 350:
        raise ValueError("Invalid model identifier")
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    if not allow_download:
        raise ModelSetupError(
            "Remote model loading disabled. Provide a local model directory "
            "or explicitly pass --allow-model-download.")
    if model_id not in MODELS.values():
        raise ModelSetupError("Only allowlisted model repositories may be downloaded")
    return model_id


def run_model(
    model: str, wav_path: Path, *, duration_ms: int,
    device: str = "cpu", model_id: str | None = None,
    allow_download: bool = False, pipeline: Any = None,
    diarizen_embedding_model: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the entire single-channel frozen WAV in one offline inference.

    The injectable pipeline is exclusively for deterministic tests. It must
    follow the third-party callable protocol and is not a model-quality proxy.
    """
    if model not in MODELS or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported Phase 1B model or device")
    wav_path = Path(wav_path).resolve(strict=True)
    if not 100 <= duration_ms <= 3600000:
        raise ValueError("Invalid source duration")
    configured = model_id or MODELS[model]
    started = time.perf_counter()
    if pipeline is None:
        configured = _validate_model_path(configured, allow_download)
        if model == "pyannote":
            try:
                from pyannote.audio import Pipeline
                token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                kwargs = {"token": token} if token else {}
                pipeline = Pipeline.from_pretrained(configured, **kwargs)
                if pipeline is None:
                    raise ModelSetupError("Unable to load pyannote Community-1")
                if device == "cuda":
                    import torch
                    pipeline.to(torch.device("cuda"))
            except (ImportError, OSError, RuntimeError) as exc:
                raise ModelSetupError(
                    "pyannote Community-1 is unavailable. Use a separate "
                    "environment and accept its Hugging Face conditions."
                ) from exc
        else:
            try:
                from diarizen.pipelines.inference import DiariZenPipeline
                if Path(configured).is_dir():
                    if (not diarizen_embedding_model or
                            not Path(diarizen_embedding_model).is_file()):
                        raise ModelSetupError(
                            "Offline DiariZen weights also require the local "
                            "Wespeaker embedding .bin file.")
                    pipeline = DiariZenPipeline(
                        diarizen_hub=Path(configured),
                        embedding_model=str(Path(diarizen_embedding_model).resolve()),
                    )
                else:
                    pipeline = DiariZenPipeline.from_pretrained(configured)
                if hasattr(pipeline, "to"):
                    import torch
                    if device == "cuda" and not torch.cuda.is_available():
                        raise ModelSetupError("CUDA unavailable for DiariZen")
                    pipeline.to(torch.device(device))
            except (ImportError, OSError, RuntimeError) as exc:
                raise ModelSetupError(
                    "DiariZen needs its separate Python/Torch environment; "
                    "the optional model weights are non-commercial."
                ) from exc
    try:
        result = pipeline(str(wav_path))
    except Exception as exc:
        raise ModelSetupError(f"{model} offline inference failed") from exc
    turns = _normalize_turns(result, duration_ms)
    elapsed = time.perf_counter() - started
    return turns, {
        "model": model, "model_id": configured,
        "device": device, "inference_mode": "offline_full_recording",
        "wall_seconds_including_load": round(elapsed, 3),
        "turns": len(turns),
        "predicted_speakers": len({t["speaker_id"] for t in turns}),
        "weight_license": (
            "CC-BY-NC-4.0 research only" if model == "diarizen"
            else "see official pyannote model card"
        ),
    }

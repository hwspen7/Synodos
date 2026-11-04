"""Optional, replaceable speaker-model adapters.

Identity verification (embeddings) and who-spoke-when diarization are different
tasks. These adapters are *comparison-only*: production still uses CAM++ and
the existing cluster backend unless a later, evidence-backed migration is
explicitly approved. No model loads or downloads during module import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import numpy as np

SAMPLE_RATE = 16_000
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
NEMOTRON_MODEL = "nvidia/Nemotron-3-Diarization"


class SpeakerModelUnavailable(RuntimeError):
    """An explicitly requested optional model cannot run."""


class EmbeddingBackend(Protocol):
    name: str
    model_id: str

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Return a unit-length vector, or None for unusable input."""


class DiarizationBackend(Protocol):
    name: str
    model_id: str

    def diarize(self, audio: np.ndarray) -> list[dict]:
        """Return anonymous speaker turns in absolute milliseconds."""


def validated_audio(audio: np.ndarray, *, max_seconds: int = 180) -> np.ndarray:
    """Require an explicitly supplied, finite, 16 kHz mono float waveform."""
    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise ValueError("Speaker models require mono 16 kHz samples")
    if samples.size == 0 or samples.size > SAMPLE_RATE * max_seconds:
        raise ValueError("Invalid or excessive speaker-model audio length")
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError("Non-numeric speaker-model audio")
    samples = samples.astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite speaker-model audio")
    if np.max(np.abs(samples)) > 1.001:
        raise ValueError("Expected normalized float audio in [-1, 1]")
    return np.ascontiguousarray(samples)


def unit_embedding(embedding: object) -> np.ndarray | None:
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if not 32 <= vec.size <= 4096 or not np.isfinite(vec).all():
        return None
    magnitude = float(np.linalg.norm(vec))
    if not math.isfinite(magnitude) or magnitude < 1e-7:
        return None
    return np.ascontiguousarray(vec / magnitude)


@dataclass
class CamPlusEmbedding:
    """Wrap the exact production CAM++ manager, keeping its scheduler/lock."""

    manager: object
    name: str = "camplus"
    model_id: str = "production/funasr-cam++"

    def __post_init__(self) -> None:
        configured = getattr(getattr(self.manager, "settings", None),
                             "spk_model", None)
        if configured:
            self.model_id = str(configured)

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        vector = self.manager.embed(validated_audio(audio))
        return unit_embedding(vector)


class EcapaEmbedding:
    """SpeechBrain ECAPA-TDNN, loaded only in an explicitly invoked comparison.

    Run in a dedicated environment when its Torch dependencies differ from
    the tested live Whisper stack. Never compare its vector to a CAM++ vector.
    """

    name = "ecapa"
    model_id = ECAPA_MODEL

    def __init__(self, *, device: str = "cpu", classifier=None) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("ECAPA device must be cpu or cuda")
        self.device = device
        self._classifier = classifier

    def _load(self):
        if self._classifier is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                self._classifier = EncoderClassifier.from_hparams(
                    source=self.model_id, run_opts={"device": self.device},
                )
            except (ImportError, OSError, RuntimeError) as exc:
                raise SpeakerModelUnavailable(
                    "ECAPA requires SpeechBrain and its model weights in the "
                    "dedicated speaker comparison environment"
                ) from exc
        return self._classifier

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        samples = validated_audio(audio)
        # Reject near silence instead of inventing a speaker prototype.
        if float(np.sqrt(np.mean(samples * samples))) < 1e-5:
            return None
        try:
            import torch
        except ImportError as exc:
            raise SpeakerModelUnavailable("ECAPA requires PyTorch") from exc
        with torch.inference_mode():
            tensor = torch.from_numpy(samples).unsqueeze(0).to(self.device)
            embedding = self._load().encode_batch(tensor)
        return unit_embedding(embedding.detach().float().cpu().numpy())


def normalize_nemotron_turns(
    raw: object, *, duration_ms: int, max_turns: int = 10000,
) -> list[dict]:
    """Convert NVIDIA's anonymous channel indexes to positive local IDs."""
    if not isinstance(raw, list) or len(raw) > max_turns:
        raise ValueError("Invalid Nemotron speaker turn list")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Malformed Nemotron speaker turn")
        start, end, slot = item.get("Start"), item.get("End"), item.get("Speaker")
        if (isinstance(slot, (bool, str))
                or not isinstance(slot, (int, np.integer)) or not 0 <= slot < 8
                or isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (float, int, np.floating, np.integer))
                or not isinstance(end, (float, int, np.floating, np.integer))
                or not math.isfinite(start) or not math.isfinite(end)):
            raise ValueError("Invalid Nemotron speaker turn values")
        a, b = round(start * 1000), round(end * 1000)
        if a < 0 or b <= a or a >= duration_ms or b > duration_ms + 20:
            raise ValueError("Nemotron turn lies outside the source audio")
        result.append({
            "start_ms": a,
            "end_ms": min(b, duration_ms),
            "speaker_id": slot + 1,
        })
    return sorted(result, key=lambda item: (
        item["start_ms"], item["end_ms"], item["speaker_id"],
    ))


class Nemotron3Diarization:
    """Transformers offline inference, for paired comparison only.

    A future realtime adapter must maintain the streaming speaker cache
    across chunks. Re-running this offline function on rolling windows would
    silently reset anonymous speaker identities and is forbidden.
    """

    name = "nemotron3"
    model_id = NEMOTRON_MODEL

    def __init__(self, *, device: str = "cpu", model=None, processor=None):
        if device not in ("cpu", "cuda"):
            raise ValueError("Nemotron device must be cpu or cuda")
        self.device = device
        self._model = model
        self._processor = processor

    def _load(self):
        if self._model is None or self._processor is None:
            try:
                from transformers import (
                    AutoModelForAudioFrameClassification, AutoProcessor,
                )
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForAudioFrameClassification.from_pretrained(
                    self.model_id,
                ).to(self.device)
                model.eval()
            except (ImportError, OSError, RuntimeError) as exc:
                raise SpeakerModelUnavailable(
                    "Nemotron 3 requires a compatible Transformers release, "
                    "PyTorch and downloaded NVIDIA model weights. Install "
                    "these in an isolated comparison environment."
                ) from exc
            self._processor, self._model = processor, model
        return self._model, self._processor

    def diarize(self, audio: np.ndarray) -> list[dict]:
        samples = validated_audio(audio)
        try:
            import torch
        except ImportError as exc:
            raise SpeakerModelUnavailable("Nemotron 3 requires PyTorch") from exc
        model, processor = self._load()
        sample_rate = processor.feature_extractor.sampling_rate
        if sample_rate != SAMPLE_RATE:
            raise SpeakerModelUnavailable("Unexpected Nemotron sampling rate")
        inputs = processor(samples, sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        turns = processor.extract_speaker_dict(
            logits, inputs.attention_mask,
        )[0]
        return normalize_nemotron_turns(
            turns, duration_ms=round(len(samples) * 1000 / SAMPLE_RATE),
        )


def create_embedding_backend(name: str, *, manager=None,
                             device: str = "cpu") -> EmbeddingBackend:
    if name == "camplus":
        if manager is None:
            raise ValueError("CAM++ requires the existing ModelManager")
        return CamPlusEmbedding(manager)
    if name == "ecapa":
        return EcapaEmbedding(device=device)
    raise ValueError(f"Unknown embedding model: {name}")


def create_diarization_backend(name: str, *, device: str = "cpu") -> DiarizationBackend:
    if name == "nemotron3":
        return Nemotron3Diarization(device=device)
    raise ValueError(f"Unknown diarization model: {name}")

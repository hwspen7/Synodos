"""Opt-in ECAPA speaker shadow worker (separate Python environment).

The protocol is newline JSON over private stdio. No waveform, speaker
embedding or identity is saved to disk. The only output is a coarse rank
comparison, NOT a calibrated probability or identity decision.
"""
from __future__ import annotations

import base64
import json
import sys

import numpy as np

from speech.speaker_backends import EcapaEmbedding, unit_embedding

PREFIX = "MEETING_ECAPA_SHADOW_V1 "
MAX_AUDIO_SAMPLES = 16_000 * 6
MAX_LABELS = 24


class ShadowCentroids:
    """In-memory ECAPA prototypes anchored to *unverified* CAM++ labels.

    Rank agreement is a diagnostic of model consistency, never proof that
    CAM++ is correct. An ECAPA conflict must not modify the baseline labels.
    """

    def __init__(self):
        self._centers: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

    def observe(self, label: int, embedding: object) -> str:
        if type(label) is not int or label <= 0:
            return "unlabeled"
        vec = unit_embedding(embedding)
        if vec is None:
            return "no_embedding"
        if label not in self._centers:
            if len(self._centers) >= MAX_LABELS:
                return "capacity"
            self._centers[label] = vec.copy()
            self._counts[label] = 1
            return "seeded"
        # One prototype alone is not an independent identity verification.
        if len(self._centers) == 1:
            self._add(label, vec)
            return "no_comparator"
        scores = {identity: float(np.dot(center, vec))
                  for identity, center in self._centers.items()}
        winner = max(sorted(scores), key=lambda identity: scores[identity])
        if winner != label:
            # Conflict must never contaminate a reference prototype.
            return "rank_conflict"
        self._add(label, vec)
        return "rank_match"

    def _add(self, label: int, vec: np.ndarray) -> None:
        n = self._counts[label]
        mixed = self._centers[label] * n + vec
        normalized = unit_embedding(mixed)
        if normalized is not None:
            self._centers[label] = normalized
            self._counts[label] += 1


def decode_audio(payload: dict) -> np.ndarray:
    if type(payload.get("sample_rate")) is not int or payload["sample_rate"] != 16000:
        raise ValueError("Unsupported sample rate")
    encoded = payload.get("pcm16_b64")
    if not isinstance(encoded, str) or len(encoded) > MAX_AUDIO_SAMPLES * 3:
        raise ValueError("Invalid PCM16 payload")
    raw = base64.b64decode(encoded, validate=True)
    if (len(raw) % 2 or not 32000 <= len(raw) <= MAX_AUDIO_SAMPLES * 2):
        raise ValueError("Invalid PCM16 length")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("cpu", "cuda"):
        return 2
    device = sys.argv[1]
    backend = EcapaEmbedding(device=device)
    centers = ShadowCentroids()
    for line in sys.stdin:
        try:
            if len(line) > MAX_AUDIO_SAMPLES * 3 + 512:
                raise ValueError("Oversized request")
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("Invalid request")
            if message.get("op") == "stop":
                break
            if message.get("op") != "observe":
                raise ValueError("Unknown request")
            label = message.get("label")
            samples = decode_audio(message)
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms < 0.008:
                outcome = "low_energy"
            else:
                vector = backend.embed(samples)
                outcome = centers.observe(label, vector)
        except Exception:
            # Do not print exception details, raw audio, or model output.
            outcome = "error"
        sys.stdout.write(PREFIX + json.dumps({"outcome": outcome}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

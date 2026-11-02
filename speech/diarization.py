"""Cross-language speaker ID stability tools, separate from ASR language detection."""
from __future__ import annotations
import numpy as np


class SpeakerCentroidTracker:
    """Incremental cosine similarity matching, independent of spoken language."""
    def __init__(self, threshold: float = 0.72, max_speakers: int = 64):
        if not 0 < threshold < 1 or max_speakers <= 0:
            raise ValueError("Invalid speaker tracker limits")
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}

    def assign(self, embedding: np.ndarray | None) -> int | None:
        if embedding is None:
            return None
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        vec = vec / norm
        similarities = [(float(np.dot(vec, c)), sid) for sid, c in self.centroids.items() if c.shape == vec.shape]
        if similarities and max(similarities)[0] >= self.threshold:
            sid = max(similarities)[1]
            n = self.counts[sid]
            combined = self.centroids[sid] * n + vec
            self.centroids[sid] = combined / max(np.linalg.norm(combined), 1e-6)
            self.counts[sid] += 1
            return sid
        if len(self.centroids) >= self.max_speakers:
            return None
        sid = max(self.centroids, default=0) + 1
        self.centroids[sid] = vec
        self.counts[sid] = 1
        return sid

    def seed(self, speaker_id: int, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if speaker_id < 1 or not np.isfinite(norm) or norm < 1e-6:
            return
        if speaker_id not in self.centroids:
            self.centroids[speaker_id] = vec / norm
            self.counts[speaker_id] = 1

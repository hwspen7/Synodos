"""A speaker embedding timeline independent of ASR sentence boundaries.

Voice embeddings are extracted from continuous acoustic windows, then assigned
to text by absolute sample overlap. These are evidence-based assignments, not
guesses from transcript wording or neighboring labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def normalize(raw) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        v = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if not v.size or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-7 else None


@dataclass(frozen=True)
class AcousticWindow:
    start_sample: int
    end_sample: int
    embedding: np.ndarray


class AcousticTimeline:
    def __init__(self, sample_rate: int, max_windows: int = 320):
        if sample_rate < 1000 or max_windows < 2:
            raise ValueError("Invalid speaker timeline bounds")
        self.sample_rate = sample_rate
        self.max_windows = max_windows
        self.windows: list[AcousticWindow] = []

    def add(self, start: int, end: int, raw_embedding) -> bool:
        embedding = normalize(raw_embedding)
        if embedding is None or end <= start or start < 0:
            return False
        if any(w.start_sample == start and w.end_sample == end for w in self.windows):
            return False
        self.windows.append(AcousticWindow(start, end, embedding))
        self.windows.sort(key=lambda w: w.start_sample)
        if len(self.windows) > self.max_windows:
            del self.windows[: len(self.windows) - self.max_windows]
        return True

    def embedding_for(
        self, start: int, end: int, *,
        min_coverage: float = 0.65,
        min_consistency: float = 0.58,
    ) -> np.ndarray | None:
        """Return a consensus embedding only when overlapping windows agree.

        Requiring acoustic support over the segment prevents assigning speakers
        solely because a neighboring sentence has an identity.
        """
        if start < 0 or end <= start:
            return None
        relevant: list[tuple[int, AcousticWindow]] = []
        intervals: list[tuple[int, int]] = []
        for window in self.windows:
            overlap_a = max(start, window.start_sample)
            overlap_b = min(end, window.end_sample)
            overlap = overlap_b - overlap_a
            if overlap <= 0:
                continue
            if overlap < min(self.sample_rate // 6, end - start):
                continue
            relevant.append((overlap, window))
            intervals.append((overlap_a, overlap_b))
        if not relevant:
            return None
        intervals.sort()
        covered = 0
        last = start
        for a, b in intervals:
            if b > last:
                covered += b - max(a, last)
                last = b
        if covered / (end - start) < min_coverage:
            return None
        dim = relevant[0][1].embedding.size
        if any(w.embedding.size != dim for _, w in relevant):
            return None
        vectors = np.stack([w.embedding for _, w in relevant])
        weights = np.array([overlap for overlap, _ in relevant], dtype=np.float64)
        weights /= weights.sum()
        mean = normalize(np.sum(vectors * weights[:, None], axis=0))
        if mean is None:
            return None
        if len(relevant) > 1 and float(np.min(vectors @ mean)) < min_consistency:
            return None
        return mean

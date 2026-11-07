"""Recover a consistent voice from a multipart ASR utterance.

A short or missing subturn embedding is *missing evidence*, not proof that
the entire multi-second sentence has no voiceprint. Require multiple strong
consistent parts, high time coverage and no contradictory observed voice.
All original ASR words and source timestamps remain canonical.
"""
from __future__ import annotations

import numpy as np


def _unit(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if vector.size < 2 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-7 else None


def consistent_partition_embedding(
    pieces: object, *,
    similarity_threshold: float = 0.82,
    minimum_piece_ms: int = 800,
    minimum_supported_ms: int = 2000,
    minimum_coverage: float = 0.75,
    max_unknown_piece_ms: int = 1200,
) -> np.ndarray | None:
    """Return a single-person embedding only when subturn evidence agrees.

    At least two reliable pieces must agree. A missing short probe can be
    ignored only when sufficiently covered by reliable speech. Any strong
    contrary embedding, long unknown interval or ambiguous long piece
    forces abstention, not an invented speaker identity.
    """
    if (not isinstance(pieces, (list, tuple)) or not 2 <= len(pieces) <= 12
            or not 0.6 <= similarity_threshold <= 1
            or not 200 <= minimum_piece_ms <= 3000
            or not 1000 <= minimum_supported_ms <= 12000
            or not 0.5 <= minimum_coverage <= 0.98
            or not 200 <= max_unknown_piece_ms <= 2500):
        return None
    supported: list[tuple[int, np.ndarray]] = []
    weak: list[np.ndarray] = []
    total_ms = 0
    valid_ms = 0
    dimension: int | None = None
    last_end = -1
    for piece in pieces:
        if (not isinstance(piece, dict)
                or type(piece.get("start")) is not int
                or type(piece.get("end")) is not int
                or piece["end"] <= piece["start"]
                or (last_end >= 0 and piece["start"] < last_end)):
            return None
        duration = piece["end"] - piece["start"]
        last_end = piece["end"]
        total_ms += duration
        vector = _unit(piece.get("emb"))
        if vector is not None:
            if dimension is None:
                dimension = vector.size
            elif vector.size != dimension:
                return None
        is_uncertain = bool(piece.get("ambiguous") or piece.get("short_probe"))
        if not is_uncertain and duration >= minimum_piece_ms and vector is not None:
            supported.append((duration, vector))
            valid_ms += duration
        else:
            # A long unsupported or explicitly ambiguous voice could be an
            # entirely different speaker; do not quietly absorb it.
            if duration > max_unknown_piece_ms:
                return None
            if vector is not None:
                weak.append(vector)

    if (len(supported) < 2 or valid_ms < minimum_supported_ms
            or total_ms <= 0 or valid_ms / total_ms < minimum_coverage):
        return None

    vectors = [item for _, item in supported]
    # Pairwise check stops one good anchor masking conflicting speakers.
    if any(
        float(np.dot(vectors[i], vectors[j])) < similarity_threshold
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ):
        return None
    weights = np.asarray([duration for duration, _ in supported], dtype=np.float64)
    weights /= weights.sum()
    consensus = _unit(np.sum(
        np.stack(vectors, axis=0) * weights[:, None], axis=0
    ))
    if consensus is None:
        return None
    if any(float(np.dot(consensus, vector)) < similarity_threshold
           for vector in weak):
        return None
    return consensus

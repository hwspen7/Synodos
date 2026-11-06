"""Determine whether optional acoustic subturns support a single voice.

ASR text is canonical. Two acoustic windows do not imply two human speakers.
This helper can recover a usable voice embedding without rewriting text,
but only when all word-conserving parts have strong independent agreement.
"""
from __future__ import annotations

import numpy as np


def consistent_partition_embedding(
    pieces: object, *,
    similarity_threshold: float = 0.82,
    minimum_piece_ms: int = 800,
) -> np.ndarray | None:
    if (not isinstance(pieces, (list, tuple))
            or len(pieces) < 2 or len(pieces) > 12
            or not 0.6 <= similarity_threshold <= 1
            or not 200 <= minimum_piece_ms <= 3000):
        return None
    vectors = []
    for piece in pieces:
        if (not isinstance(piece, dict) or piece.get("ambiguous")
                or piece.get("short_probe")
                or type(piece.get("start")) is not int
                or type(piece.get("end")) is not int
                or piece["end"] - piece["start"] < minimum_piece_ms):
            return None
        try:
            vec = np.asarray(piece.get("emb"), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if vec.size < 2 or not np.isfinite(vec).all():
            return None
        length = float(np.linalg.norm(vec))
        if length <= 1e-7:
            return None
        vectors.append(vec / length)
    if any(v.shape != vectors[0].shape for v in vectors):
        return None
    if any(
        float(np.dot(vectors[0], v)) < similarity_threshold
        for v in vectors[1:]
    ):
        return None
    combined = np.mean(vectors, axis=0)
    length = float(np.linalg.norm(combined))
    return combined / length if length > 1e-7 else None

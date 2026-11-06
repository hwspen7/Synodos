"""Preserve recognized speech when a voiceprint embedding is unavailable."""

from __future__ import annotations

import numpy as np

UNIDENTIFIED_CLUSTER = -3


def normalized_embedding(value: object) -> np.ndarray | None:
    """Return a usable speaker vector or None without losing the ASR sentence."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr).all() or float(np.linalg.norm(arr)) <= 1e-8:
        return None
    return arr


def fallback_speaker_id(segment_id: int) -> int:
    """Use a stable, unique non-positive ID for each unidentified sentence."""
    return -max(1, int(segment_id))

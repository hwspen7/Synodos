"""Conservative matching of enrolled voices within one embedding model.

Cosine similarity is not a calibrated identity probability. A match must
exceed the configured threshold and be separated from the runner-up
identity. Different embedding models or dimensions must never be compared.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def _unit(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, OverflowError):
        return None
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        return None
    size = float(np.linalg.norm(vector))
    if not math.isfinite(size) or size <= 1e-8:
        return None
    return vector / size


def match_enrolled_voice(
    participants: list[dict], embedding: Any, *,
    model_id: str = "camplus",
    threshold: float = 0.3,
    margin: float = 0.06,
) -> dict:
    """Return a raw-cosine candidate or abstain when evidence is ambiguous."""
    if not -1.0 <= threshold <= 1.0 or not 0 <= margin <= 2:
        raise ValueError("Invalid voiceprint match thresholds")
    probe = _unit(embedding)
    if probe is None:
        return {"status": "invalid_probe", "profile_id": None,
                "similarity": None, "margin": None}
    ranked: list[tuple[float, str]] = []
    for person in participants:
        candidate_scores: list[float] = []
        for row in person.get("vectors") or []:
            if row.get("embedding_model", "camplus") != model_id:
                continue
            if bool(row.get("short")):
                continue
            enrolled = _unit(row.get("emb"))
            if enrolled is None or enrolled.size != probe.size:
                continue
            candidate_scores.append(float(np.dot(probe, enrolled)))
        if candidate_scores:
            ranked.append((max(candidate_scores), str(person["id"])))
    ranked.sort(reverse=True, key=lambda v: v[0])
    if not ranked:
        return {"status": "no_compatible_enrollment", "profile_id": None,
                "similarity": None, "margin": None}
    top, sid = ranked[0]
    delta = top - ranked[1][0] if len(ranked) > 1 else None
    if top < threshold:
        status = "below_threshold"
    elif delta is not None and delta < margin:
        status = "ambiguous"
    else:
        status = "matched"
    return {
        "status": status,
        "profile_id": sid if status == "matched" else None,
        "similarity": round(top, 6),
        "margin": round(delta, 6) if delta is not None else None,
        "score_kind": "raw_cosine_not_probability",
        "embedding_model": model_id,
    }

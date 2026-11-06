"""Conservative cross-meeting voiceprint matching.

Speaker embeddings are personal biometric data. Match only the local library,
never transmit vectors. Reject incompatible vectors, ties, short recordings,
and weak evidence instead of guessing a real person's name.
"""
from __future__ import annotations

import numpy as np


def match_saved_voice(
    embedding: np.ndarray,
    profiles: list[dict],
    *,
    minimum_similarity: float = 0.78,
    minimum_margin: float = 0.08,
) -> tuple[str | None, float]:
    """Return a stable profile ID only when one candidate clearly wins."""
    probe = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(probe))
    if not np.isfinite(probe).all() or not np.isfinite(norm) or norm < 1e-8:
        return None, -1.0
    probe = probe / norm
    candidates: list[tuple[float, str]] = []
    for profile in profiles:
        scores = []
        for vector in profile.get("vectors", []):
            if vector.get("embedding_model", "camplus") != "camplus":
                continue
            if vector.get("short") or float(vector.get("duration_s", 0)) < 2:
                continue
            value = np.asarray(vector.get("emb"), dtype=np.float32).reshape(-1)
            if value.shape != probe.shape or not np.isfinite(value).all():
                continue
            length = float(np.linalg.norm(value))
            if not np.isfinite(length) or length < 1e-8:
                continue
            scores.append(float(np.dot(probe, value / length)))
        if scores:
            candidates.append((max(scores), str(profile["id"])))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates or candidates[0][0] < minimum_similarity:
        return None, candidates[0][0] if candidates else -1.0
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < minimum_margin:
        return None, candidates[0][0]
    return candidates[0][1], candidates[0][0]


def resolved_speaker_identity(segment: dict, anonymous_id: int | None) -> tuple[int, str, str]:
    """Reconcile anonymous diarization with a verified library match.

    Manual corrections always win. A defensible registered voice identity
    wins over an anonymous diarization number; uncertainty never fabricates
    a real-world identity.
    """
    if segment.get("speaker_override"):
        sid = int(segment.get("speaker_override_id") or -1)
        name = str(segment.get("speaker_override_name") or "Unidentified Speaker")
        return sid, name, "manual_override"
    profile_id = segment.get("matched_profile_id")
    profile_name = segment.get("matched_profile_name")
    if profile_id and isinstance(profile_name, str) and profile_name.strip():
        sid = int(segment.get("speaker_id") or -1)
        if sid > 0:
            return sid, profile_name, "registered_voiceprint"
    if anonymous_id is not None:
        return (
            anonymous_id,
            f"Speaker {anonymous_id}" if anonymous_id > 0 else "Unidentified Speaker",
            "nemotron3" if anonymous_id > 0 else "nemotron3_uncertain",
        )
    sid = int(segment.get("speaker_id") or -1)
    return sid, str(segment.get("speaker_name") or "Unidentified Speaker"), "camplus"

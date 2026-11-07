"""Nonprobabilistic per-caption acoustic speaker evidence for later summaries.

Cosine similarity describes embedding geometry and MUST NOT be advertised
as a calibrated probability of speaker correctness. No label is created
or reassigned here; this is presentation metadata over existing IDs.
"""
from __future__ import annotations

import numpy as np


def _unit(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.size < 2 or not np.isfinite(arr).all():
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if np.isfinite(n) and n > 1e-7 else None


def speaker_evidence(segments: list[dict]) -> dict[int, dict]:
    """Derive bounded raw similarity and a conservative *status*.

    A provisional single-voice identity is not silently upgraded to
    confirmed. Similarity is compared to other samples of the existing
    published identity, not to a human ground-truth voice.
    """
    # Maintain per-person, per-embedding-dimension accumulators. This
    # makes a snapshot linear in the transcript length and cannot fail
    # when an old saved meeting used a different embedding model.
    groups: dict[tuple[int, int], dict] = {}
    embeddings: dict[int, np.ndarray | None] = {}
    for segment in segments:
        sid = int(segment["id"])
        vector = _unit(segment.get("emb"))
        embeddings[sid] = vector
        person = segment.get("speaker_id")
        if type(person) is int and person > 0 and vector is not None:
            key = (person, vector.size)
            entry = groups.setdefault(key, {
                "count": 0,
                "sum": np.zeros(vector.size, dtype=np.float64),
            })
            entry["count"] += 1
            entry["sum"] += vector

    centers = {
        key: _unit(entry["sum"])
        for key, entry in groups.items()
    }
    results = {}
    for seg in segments:
        sid = int(seg["id"])
        person = seg.get("speaker_id")
        vector = embeddings[sid]
        key = (person, vector.size) if (
            type(person) is int and vector is not None
        ) else None
        own = groups.get(key) if key is not None else None
        support = own["count"] if own is not None else 0
        raw_score = None
        if vector is not None and support >= 2:
            # Exclude this utterance from its own comparison template.
            center = _unit(own["sum"] - vector)
            if center is not None:
                raw_score = float(np.dot(vector, center))
        enrolled_score = seg.get("speaker_match_similarity")
        if (seg.get("cluster") == -1
                and isinstance(enrolled_score, (int, float))
                and np.isfinite(enrolled_score)):
            raw_score = float(enrolled_score)
        alternatives = []
        if vector is not None:
            for (other_person, dimension), center in centers.items():
                if (other_person == person or dimension != vector.size
                        or center is None):
                    continue
                alternatives.append(float(np.dot(vector, center)))
        margin = raw_score - max(alternatives) if (
            raw_score is not None and alternatives
        ) else None
        if seg.get("speaker_override"):
            status = "manual_override"
        elif type(person) is not int or person <= 0:
            status = "unidentified"
        elif seg.get("speaker_uncertain") or vector is None:
            status = "uncertain"
        elif seg.get("cluster") == -1:
            status = "enrolled_match"
        elif (support >= 2 and raw_score is not None
              and raw_score >= 0.68
              and (margin is None or margin >= 0.08)):
            status = "supported"
        else:
            status = "tentative"
        results[sid] = {
            "speaker_status": status,
            "speaker_similarity": (
                round(float(raw_score), 5)
                if raw_score is not None else None
            ),
            "speaker_margin": (
                round(float(margin), 5)
                if margin is not None else None
            ),
            "speaker_evidence_count": max(
                0, support - (1 if vector is not None else 0)
            ),
            "speaker_evidence_status": str(
                seg.get("speaker_evidence_status") or "unavailable"
            ),
            "speaker_score_kind": "raw_cosine_not_probability",
        }
    return results

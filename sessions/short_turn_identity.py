"""Conservative recovery of short turns from a previously established voice.

A <1.2-second short probe is insufficient to BIRTH another speaker, but
it may corroborate an existing speaker in the same recording. Keep this
independent of transcript language and never assign from text content.
"""
from __future__ import annotations

import numpy as np


def _unit(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        v = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if v.size < 2 or not np.isfinite(v).all():
        return None
    size = float(np.linalg.norm(v))
    return v / size if size > 1e-7 else None


def recover_short_turn_identity(
    records: list[dict], *,
    sample_rate: int,
    similarity_floor: float = 0.83,
    margin: float = 0.12,
    max_gap_seconds: float = 8.0,
) -> int:
    """Assign an uncertain SHORT probe to an already known nearby voice.

    Only a usable probe embedding and a strong, unambiguous matching voice
    are acceptable. Missing embeddings, overlap, long gaps, and an ambiguous
    match remain unidentified. This is not evidence of a newly born person.
    Existing registered/manual identities are never modified.
    """
    if (sample_rate <= 0 or not 0.5 < similarity_floor <= 1
            or not 0 <= margin <= 0.5
            or not 0 < max_gap_seconds <= 30):
        raise ValueError("Invalid short-turn identity recovery policy")
    ordered = sorted(
        records, key=lambda record: (
            int(record.get("start_sample") or 0),
            int(record.get("id") or 0),
        ),
    )
    reliable: list[tuple[int, int, np.ndarray]] = []
    fixed = 0
    for segment in ordered:
        start = int(segment.get("start_sample") or 0)
        end = int(segment.get("end_sample") or 0)
        if end <= start:
            continue
        emb = _unit(segment.get("emb"))
        sid = segment.get("speaker_id")
        short_probe = (
            segment.get("speaker_evidence_status") in (
                "short_probe", "short_probe_known_voice"
            )
            and bool(segment.get("speaker_uncertain"))
        )
        if (short_probe and emb is not None
                and not segment.get("speaker_override")
                and not segment.get("matched_profile_id")):
            # Only a trusted, earlier, nonoverlapping source can establish
            # continuity. Do not create an identity from other short probes.
            votes: dict[int, float] = {}
            for label, last_end, reference in reliable:
                gap = start - last_end
                if 0 <= gap <= int(max_gap_seconds * sample_rate) and (
                    reference.shape == emb.shape
                ):
                    score = float(np.dot(emb, reference))
                    votes[label] = max(votes.get(label, -1.0), score)
            ranked = sorted(
                votes.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if (ranked and ranked[0][1] >= similarity_floor
                    and (len(ranked) == 1
                         or ranked[0][1] - ranked[1][1] >= margin)):
                label = ranked[0][0]
                segment["speaker_id"] = label
                segment["speaker_name"] = f"Speaker {label}"
                segment["speaker_evidence_status"] = "short_probe_known_voice"
                # The short clip is still uncertain, and cannot become a
                # reference for further automatic propagation.
                segment["speaker_uncertain"] = True
                fixed += 1
            continue

        if (type(sid) is int and sid > 0
                and not segment.get("speaker_uncertain")
                and emb is not None):
            reliable.append((sid, end, emb))
            # Bound memory in long meetings without growing latency.
            if len(reliable) > 64:
                del reliable[:-64]
    return fixed

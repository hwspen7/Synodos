"""Evidence-gated anonymous speaker births for streaming CAM++ clusters.

Raw cluster IDs are acoustic partitions, not people. In particular a
complete-link clustering backend can split one changing voice into many
singletons. Only give a *new* positive speaker ID to an independently
corroborated distinct voice, while a strong voice match can reuse an
existing anonymous speaker's cluster. No text is rewritten.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


UNKNOWN_CLUSTER = -3


def _unit(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vec.size < 2 or not np.isfinite(vec).all():
        return None
    length = float(np.linalg.norm(vec))
    if length < 1e-7:
        return None
    return vec / length


def _observations(members: list[dict], sample_rate: int) -> int:
    """Count separated original utterances, not overlapping worker outputs."""
    spans = sorted(
        (int(m["start_sample"]), int(m["end_sample"]))
        for m in members if _unit(m.get("emb")) is not None
        and int(m["end_sample"]) > int(m["start_sample"])
    )
    count, previous_end = 0, -1
    for start, end in spans:
        if previous_end < 0 or start - previous_end >= int(.65 * sample_rate):
            count += 1
        previous_end = max(previous_end, end)
    return count


def gate_new_cluster_births(
    records: list[dict], *,
    sample_rate: int,
    same_voice_similarity: float = .78,
    distinct_voice_max_similarity: float = .48,
) -> dict[str, int]:
    """Conservatively remap unregistered raw clusters before ID allocation.

    A first supported voice can start Speaker 1 immediately. Later raw
    clusters reuse it only on strong acoustic agreement. Otherwise a new
    person requires two independent usable turns OR one long, precise
    word-aligned voice turn, plus clear acoustic difference to established
    voices. With weaker evidence remain unknown until later corroboration.

    Known enrolled identities and manual overrides are never changed.
    """
    if (sample_rate <= 0 or not .5 < same_voice_similarity <= 1
            or not 0 <= distinct_voice_max_similarity < same_voice_similarity):
        raise ValueError("Invalid speaker birth gate thresholds")
    groups: dict[int, list[dict]] = defaultdict(list)
    for segment in records:
        raw = segment.get("cluster")
        if (type(raw) is int and raw >= 0
                and not segment.get("speaker_override")
                and not segment.get("matched_profile_id")):
            groups[raw].append(segment)
    if not groups:
        return {"reused": 0, "deferred": 0, "confirmed": 0}
    ordered = sorted(
        groups,
        key=lambda raw: (
            min(int(s["start_sample"]) for s in groups[raw]), raw
        ),
    )
    accepted: list[tuple[int, list[np.ndarray]]] = []
    reused = deferred = confirmed = 0
    for raw in ordered:
        members = groups[raw]
        vectors = [vec for s in members if (vec := _unit(s.get("emb"))) is not None
                   and not (s.get("speaker_evidence_status") == "short_probe"
                            and s.get("speaker_uncertain"))]
        if not vectors:
            for record in members:
                record["cluster"] = UNKNOWN_CLUSTER
                record["speaker_id"] = -1
                record["speaker_name"] = "Unidentified Speaker"
                record["speaker_uncertain"] = True
                record["speaker_evidence_status"] = "speaker_birth_awaiting_voice"
            deferred += len(members)
            continue

        # Compare each source vector to each accepted cluster, rather than
        # trusting unstable clustering labels or the words being spoken.
        matches: list[tuple[float, int]] = []
        for anchor_raw, reference in accepted:
            scores = [
                max((float(np.dot(vec, ref)) for ref in reference
                     if vec.shape == ref.shape), default=-1.0)
                for vec in vectors
            ]
            if scores:
                matches.append((min(scores), anchor_raw))
        matches.sort(reverse=True)
        top = matches[0] if matches else None
        runner_up = matches[1][0] if len(matches) > 1 else -1.0
        if (top is not None and top[0] >= same_voice_similarity
                and top[0] - runner_up >= .08):
            anchor = top[1]
            anchor_members = groups[anchor]
            anchored = next(
                (s for s in anchor_members
                 if type(s.get("speaker_id")) is int and s["speaker_id"] > 0),
                None,
            )
            for record in members:
                record["cluster"] = anchor
                if anchored is not None:
                    record["speaker_id"] = anchored["speaker_id"]
                    record["speaker_name"] = anchored.get("speaker_name") or (
                        f"Speaker {anchored['speaker_id']}"
                    )
                record["speaker_evidence_status"] = "speaker_birth_voice_reused"
            reused += len(members)
            accepted = [
                (candidate, references + vectors if candidate == anchor else references)
                for candidate, references in accepted
            ]
            continue

        first = not accepted
        observations = _observations(members, sample_rate)
        long_precise = any(
            (int(s["end_sample"]) - int(s["start_sample"]))
                >= int(3.5 * sample_rate)
            and s.get("timestamp_quality") == "word"
            and len(s.get("words") or ()) >= 3
            and _unit(s.get("emb")) is not None
            for s in members
        )
        distinct = (
            first or all(
                max((float(np.dot(vec, ref)) for _, refs in accepted
                     for ref in refs if vec.shape == ref.shape), default=-1.0)
                <= distinct_voice_max_similarity
                for vec in vectors
            )
        )
        if distinct and (first or observations >= 2 or long_precise):
            accepted.append((raw, vectors))
            confirmed += 1
            continue

        for record in members:
            record["cluster"] = UNKNOWN_CLUSTER
            record["speaker_id"] = -1
            record["speaker_name"] = "Unidentified Speaker"
            record["speaker_uncertain"] = True
            record["speaker_evidence_status"] = "speaker_birth_unconfirmed"
        deferred += len(members)
    return {"reused": reused, "deferred": deferred, "confirmed": confirmed}

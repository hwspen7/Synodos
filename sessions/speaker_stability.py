"""Conservative stable labels for streaming speaker clustering.

The acoustic cluster backend is permitted to revise a recent live window.
Raw cluster numbers are not identities: map them by majority overlap with
previously published speaker IDs, and stop retroactively rewriting mature,
non-uncertain transcript segments.

This is presentation/persistence stability, NOT a claim of diarization
accuracy. A manual speaker override remains authoritative.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping


def _global_overlap_assignment(
    raw_labels: list[int],
    stable_ids: list[int],
    overlap: Counter[tuple[int, int]],
) -> dict[int, int]:
    """Maximize the total historical-label agreement across ALL raw clusters.

    A greedy pick can spend an old speaker ID on a slightly better local
    match and force a different, strongly continuing voice into a new ID.
    Exact minimum-cost assignment resolves this globally. Each raw cluster
    also has a zero-score dummy column, so weak/nonexistent matches do not
    acquire historical identities.
    """
    if not raw_labels or not stable_ids:
        return {}
    n, old_count = len(raw_labels), len(stable_ids)
    m = old_count + n
    max_weight = max(overlap.values(), default=0)
    if max_weight <= 0:
        return {}

    # Rectangular Hungarian assignment, integer costs and stable tie order.
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                weight = (
                    overlap.get((raw_labels[i0 - 1], stable_ids[j - 1]), 0)
                    if j <= old_count else 0
                )
                cost = max_weight - weight
                cur = cost - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not math.isfinite(delta):
                raise RuntimeError("No finite speaker assignment")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assigned = {}
    for j in range(1, old_count + 1):
        i = p[j]
        if i and overlap.get((raw_labels[i - 1], stable_ids[j - 1]), 0) > 0:
            assigned[raw_labels[i - 1]] = stable_ids[j - 1]
    return assigned


def stabilized_cluster_ids(
    records: Iterable[Mapping[str, object]],
    participant_labels: Iterable[int],
    old_map: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Keep globally consistent positive display IDs after reclustering.

    Raw cluster IDs are local model labels, not human identities. Historical
    segment agreement is optimized globally; manual participant labels
    remain reserved, old mappings are used only for previously unseen
    raw clusters, and retired numbers are never reassigned.
    """
    items = list(records)
    participant_set = {int(x) for x in participant_labels if int(x) > 0}
    raw_labels = sorted({
        int(s["cluster"]) for s in items
        if type(s.get("cluster")) is int and int(s["cluster"]) >= 0
    })
    reserved = set(participant_set)
    known = {
        int(s["speaker_id"]) for s in items
        if type(s.get("speaker_id")) is int and int(s["speaker_id"]) > 0
    }
    if old_map:
        known.update(int(x) for x in old_map.values() if type(x) is int and x > 0)
    reserved.update(known)

    overlap: Counter[tuple[int, int]] = Counter()
    for item in items:
        raw = item.get("cluster")
        stable = item.get("speaker_id")
        if (type(raw) is int and raw >= 0
                and type(stable) is int and stable > 0
                and stable not in participant_set
                and not item.get("speaker_override")):
            overlap[(raw, stable)] += 1

    stable_ids = sorted({
        stable for _, stable in overlap
        if stable not in participant_set
    })
    assigned = _global_overlap_assignment(raw_labels, stable_ids, overlap)
    occupied = set(participant_set) | set(assigned.values())

    if old_map:
        for raw in raw_labels:
            prior = old_map.get(raw)
            if (raw not in assigned and type(prior) is int
                    and prior > 0 and prior not in occupied):
                assigned[raw] = prior
                occupied.add(prior)

    next_id = max(reserved | occupied, default=0) + 1
    for raw in raw_labels:
        if raw not in assigned:
            assigned[raw] = next_id
            occupied.add(next_id)
            next_id += 1
    return assigned

def collect_mature_labels(
    records: Iterable[Mapping[str, object]],
    sample_rate: int,
    horizon_seconds: float,
) -> dict[int, tuple[int, str]]:
    """Remember trusted historical speaker labels before re-clustering.

    Recent and still-uncertain segments remain fully revisable. Setting
    horizon_seconds to zero disables label freezing for A/B tests.
    """
    if sample_rate <= 0 or not 0 <= horizon_seconds <= 3600:
        raise ValueError("Invalid speaker stability horizon")
    if not horizon_seconds:
        return {}
    items = list(records)
    latest = max((int(s["end_sample"]) for s in items), default=0)
    horizon = int(sample_rate * horizon_seconds)
    result = {}
    for item in items:
        sid = item.get("speaker_id")
        if (type(sid) is int and sid > 0
                and item.get("emb") is not None
                and str(item.get("speaker_name") or "").lower()
                    != "unidentified speaker"
                and not item.get("speaker_uncertain", False)
                and not item.get("speaker_override", False)
                and latest - int(item["end_sample"]) >= horizon):
            result[int(item["id"])] = (
                sid, str(item.get("speaker_name") or f"Speaker {sid}"),
            )
    return result


def restore_mature_labels(
    records: Iterable[dict],
    locked: Mapping[int, tuple[int, str]],
) -> int:
    """Keep mature historical assignments stable without touching audio."""
    restored = 0
    for item in records:
        key = int(item["id"])
        if key not in locked:
            continue
        sid, name = locked[key]
        if item.get("speaker_id") != sid or item.get("speaker_name") != name:
            restored += 1
        item["speaker_id"] = sid
        item["speaker_name"] = name
    return restored


def summarize_cluster_fragmentation(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize the final raw acoustic clusters versus published labels.

    Diagnostic only. Different raw cluster labels are not necessarily
    distinct people; two labels in one raw cluster do not prove identity.
    No transcript, voiceprint vectors or participant names are exposed.
    """
    raw_to_speaker: dict[int, Counter[int]] = {}
    speaker_to_raw: dict[int, set[int]] = {}
    labeled = set()
    without_embedding = 0
    evidence_statuses: Counter[str] = Counter()
    for record in records:
        sid = record.get("speaker_id")
        if type(sid) is int and sid > 0:
            labeled.add(sid)
        if record.get("emb") is None:
            without_embedding += 1
        reason = str(record.get("speaker_evidence_status") or "not_recorded")
        if reason not in {
            "direct_embedding", "pending_turn_analysis", "short_turn",
            "speaker_task_backpressure_or_failure",
            "turn_analysis_unavailable", "turn_voiceprint",
            "turn_ambiguous", "turn_no_voiceprint", "acoustic_backfill",
            "short_probe", "short_probe_corroborated",
            "not_recorded",
        }:
            reason = "other"
        evidence_statuses[reason] += 1
        raw = record.get("cluster")
        if type(raw) is not int or raw < 0:
            continue
        if raw not in raw_to_speaker:
            raw_to_speaker[raw] = Counter()
        if type(sid) is int and sid > 0:
            raw_to_speaker[raw][sid] += 1
            speaker_to_raw.setdefault(sid, set()).add(raw)

    conflicts = [
        {
            "raw_cluster_id": raw,
            "labeled_segments": sum(labels.values()),
            "distinct_published_labels": len(labels),
            "largest_label_segment_count": max(labels.values()),
        }
        for raw, labels in raw_to_speaker.items()
        if len(labels) > 1
    ]
    conflicts.sort(
        key=lambda item: (
            -item["distinct_published_labels"],
            -item["labeled_segments"],
            item["raw_cluster_id"],
        )
    )
    return {
        "raw_cluster_count": len(raw_to_speaker),
        "published_positive_speaker_count": len(labeled),
        "segments_without_embedding": without_embedding,
        "speaker_evidence_status_counts": dict(evidence_statuses),
        "raw_clusters_with_multiple_published_labels": len(conflicts),
        "labeled_segments_in_conflicted_raw_clusters": sum(
            item["labeled_segments"] for item in conflicts
        ),
        "max_published_labels_per_raw_cluster": max(
            (len(labels) for labels in raw_to_speaker.values()), default=0
        ),
        "published_labels_spanning_multiple_raw_clusters": sum(
            len(raws) > 1 for raws in speaker_to_raw.values()
        ),
        "largest_raw_cluster_size": max(
            (sum(labels.values()) for labels in raw_to_speaker.values()),
            default=0,
        ),
        "top_conflicted_raw_clusters": conflicts[:8],
    }


def diagnose_cluster_merge_thresholds(
    records: Iterable[Mapping[str, object]],
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.74, 0.78),
) -> dict[str, object]:
    """Read-only acoustic cluster merge sensitivity at finalization.

    Starting from the CURRENT raw clusters, greedily merge the most
    cosine-similar cluster centroids at each candidate threshold. This
    mimics FunASR's center-merging step, but does not rerun the preceding
    spectral clustering or claim a DER or speaker-count improvement.
    Never export embeddings or pairwise identity similarities.
    """
    import numpy as np

    if any(not 0 < t <= 1 for t in thresholds):
        raise ValueError("All candidate merge thresholds must be in (0, 1]")
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    dim = None
    for item in records:
        raw, emb = item.get("cluster"), item.get("emb")
        if type(raw) is not int or raw < 0 or emb is None:
            continue
        try:
            vec = np.asarray(emb, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if not vec.size or not np.isfinite(vec).all():
            continue
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 1e-8:
            continue
        if dim is None:
            dim = vec.size
        if vec.size != dim:
            continue
        sums[raw] = sums.get(raw, np.zeros(dim, dtype=np.float64)) + vec
        counts[raw] = counts.get(raw, 0) + 1
    if len(sums) > 128:
        return {"sweep_available": False,
                "reason": "more_than_128_raw_clusters"}
    if not sums:
        return {"sweep_available": False, "reason": "no_valid_embeddings"}

    def simulate(threshold: float) -> int:
        centers = {
            key: (value.copy(), counts[key]) for key, value in sums.items()
        }
        while len(centers) > 1:
            keys = sorted(centers)
            best_pair = None
            best_sim = threshold
            for i, left in enumerate(keys):
                a = centers[left][0]
                na = float(np.linalg.norm(a))
                if na <= 1e-8:
                    continue
                for right in keys[i + 1:]:
                    b = centers[right][0]
                    nb = float(np.linalg.norm(b))
                    if nb <= 1e-8:
                        continue
                    sim = float(np.dot(a, b) / (na * nb))
                    if sim >= best_sim:
                        if sim > best_sim or best_pair is None:
                            best_pair, best_sim = (left, right), sim
            if best_pair is None:
                break
            left, right = best_pair
            av, an = centers[left]
            bv, bn = centers.pop(right)
            centers[left] = (av + bv, an + bn)
        return len(centers)

    return {
        "sweep_available": True,
        "starting_raw_cluster_count": len(sums),
        "candidate_merges": [
            {
                "cosine_threshold": round(float(t), 3),
                "posthoc_cluster_count": simulate(float(t)),
            } for t in thresholds
        ],
        "method_note": (
            "Approximate post-hoc merges of CURRENT raw cluster centroids, "
            "not a fresh FunASR spectral clustering run or an identity "
            "accuracy measure."
        ),
    }


def diagnose_fresh_cluster_candidates(
    records: Iterable[Mapping[str, object]],
    thresholds: tuple[float, ...] = (0.60, 0.68, 0.72, 0.76, 0.80),
) -> dict[str, object]:
    """Actually re-cluster available voiceprints; do not merge only raw groups.

    Hypothetical counts are NOT identities or a DER measurement. In
    particular, one person with noisy embeddings may form many clusters.
    This diagnostic runs only when explicitly enabled at session closure.
    """
    import numpy as np
    from collections import Counter
    from speech.adaptive_cluster import MAX_EMBEDDINGS, complete_link_clusters

    vectors = []
    for record in records:
        if record.get("emb") is None or record.get("speaker_override"):
            continue
        try:
            vec = np.asarray(record["emb"], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            continue
        if not vec.size or not np.isfinite(vec).all():
            continue
        if vectors and vec.shape != vectors[0].shape:
            continue
        vectors.append(vec)
    if not vectors or len(vectors) > MAX_EMBEDDINGS:
        return {
            "available": False,
            "reason": "missing_or_over_256_voiceprints",
            "eligible_embeddings": len(vectors),
        }
    array = np.stack(vectors)
    candidates = []
    for threshold in thresholds:
        try:
            labels = complete_link_clusters(array, threshold=threshold)
            sizes = Counter(map(int, labels))
            candidates.append({
                "cosine_threshold": float(threshold),
                "candidate_cluster_count": len(sizes),
                "clusters_with_two_or_more_samples": sum(
                    size >= 2 for size in sizes.values()
                ),
                "singleton_clusters": sum(size == 1 for size in sizes.values()),
                "largest_cluster_size": max(sizes.values()),
            })
        except ValueError:
            candidates.append({
                "cosine_threshold": float(threshold),
                "candidate_cluster_count": None,
                "reason": "invalid_or_excessive_fragmentation",
            })
    from speech.spherical_cluster import spherical_speaker_candidates
    try:
        spherical = spherical_speaker_candidates(array)
        spherical_report = spherical.details
    except ValueError as exc:
        spherical_report = {
            "available": False,
            "reason": "spherical_validation_" + type(exc).__name__,
        }
    return {
        "available": True,
        "eligible_embeddings": len(vectors),
        "candidate_reclustering": candidates,
        "spherical_candidate": spherical_report,
        "warning": (
            "Complete-link threshold sweeps can overfragment one voice. "
            "Spherical candidates choose k via a heuristic silhouette score, "
            "not oracle person count. Neither is speaker identity ground "
            "truth or a calibrated DER improvement. Verify held-out RTTM."
        ),
    }


def anonymous_speaker_timeline(
    records: Iterable[Mapping[str, object]],
    sample_rate: int,
    *,
    max_segments: int = 256,
) -> dict[str, object]:
    """Bounded, transcript-free speaker time intervals for offline review.

    Raw cluster numbers are local to one clustering snapshot. This diagnostic
    must never contain speaker names, spoken text or voice embeddings.
    """
    if sample_rate < 1000 or not 1 <= max_segments <= 512:
        raise ValueError("Invalid anonymous speaker timeline limits")
    items = sorted(
        list(records),
        key=lambda x: (int(x.get("start_sample", 0)), int(x.get("id", 0))),
    )
    result: list[dict[str, object]] = []
    for item in items[:max_segments]:
        start = int(item.get("start_sample", 0))
        end = int(item.get("end_sample", 0))
        if start < 0 or end < start:
            continue
        raw = item.get("cluster")
        sid = item.get("speaker_id")
        result.append({
            "segment_id": int(item.get("id", 0)),
            "start_ms": round(start * 1000 / sample_rate),
            "end_ms": round(end * 1000 / sample_rate),
            "raw_cluster_id": raw if type(raw) is int and raw >= 0 else None,
            "published_speaker_id": sid if type(sid) is int and sid > 0 else None,
            "embedding_available": item.get("emb") is not None,
            "speaker_uncertain": bool(item.get("speaker_uncertain", False)),
            # Provenance only, not recognized words or recovered identity.
            # Distinct final caption segment IDs let the ASR retry auditor
            # distinguish published segments from repeated cycle candidates.
            "recovery_origin": item.get("recognition_origin") == "gap_recovery",
            "speaker_evidence_status": str(
                item.get("speaker_evidence_status") or "not_recorded"
            ),
            # Non-destructive speaker activity. It is NOT the full ASR
            # caption span and is not silently substituted into standard
            # published-caption DER.
            "speaker_activity_intervals_ms": (
                item.get("speaker_activity_intervals_ms", [])
                if isinstance(item.get("speaker_activity_intervals_ms", []), list)
                else []
            ),
            "caption_projection_status": str(
                item.get("caption_projection_status") or "disabled"
            ),
            # Only time/anonymous identity, never recognized words.
            "caption_projection_turns": [
                {
                    "start_ms": cue["start_ms"],
                    "end_ms": cue["end_ms"],
                    "speaker_id": cue["speaker_id"],
                    "word_count": cue["word_count"],
                }
                for cue in item.get("caption_cues", [])
                if isinstance(cue, dict)
                and all(key in cue for key in (
                    "start_ms", "end_ms", "speaker_id", "word_count",
                ))
            ],
            "overlap_intervals_ms": [
                dict(span) for span in item.get("overlap_intervals_ms", [])
                if isinstance(span, dict)
            ],
        })
    return {
        "total_segments": len(items),
        "exported_segments": len(result),
        "truncated": len(items) > max_segments,
        "time_reference": "Absolute milliseconds from start of received audio",
        "segments": result,
    }

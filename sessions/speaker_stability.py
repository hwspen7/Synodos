"""Conservative stable labels for streaming speaker clustering.

The acoustic cluster backend is permitted to revise a recent live window.
Raw cluster numbers are not identities: map them by majority overlap with
previously published speaker IDs, and stop retroactively rewriting mature,
non-uncertain transcript segments.

This is presentation/persistence stability, NOT a claim of diarization
accuracy. A manual speaker override remains authoritative.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping


def stabilized_cluster_ids(
    records: Iterable[Mapping[str, object]],
    participant_labels: Iterable[int],
    old_map: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Map raw cluster labels to prior stable IDs using global overlap.

    Unlike sorted-raw greedy matching, a tiny split-off raw cluster cannot
    steal a stable ID away from the larger continuing speaker cluster.
    Previously allocated IDs are never recycled while historical segments
    still refer to them.
    """
    items = list(records)
    participant_set = {int(x) for x in participant_labels if int(x) > 0}
    raw_labels = sorted({
        int(s["cluster"]) for s in items
        if isinstance(s.get("cluster"), int) and int(s["cluster"]) >= 0
    })
    reserved = set(participant_set)
    known = {
        int(s["speaker_id"]) for s in items
        if isinstance(s.get("speaker_id"), int) and int(s["speaker_id"]) > 0
    }
    if old_map:
        known.update(int(x) for x in old_map.values() if int(x) > 0)
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
    # Largest continuing clusters claim their historical identity before
    # small split-offs. Deterministic tie breaking prevents random map flips.
    ordered = sorted(
        overlap.items(),
        key=lambda item: (-item[1], item[0][1], item[0][0]),
    )
    assigned = {}
    occupied = set(participant_set)
    for (raw, stable), count in ordered:
        if count > 0 and raw not in assigned and stable not in occupied:
            assigned[raw] = stable
            occupied.add(stable)

    # A raw ID can survive a turn where its old segments are absent.
    # Reuse the old mapping only if it cannot conflict with present anchors.
    if old_map:
        for raw in raw_labels:
            prior = old_map.get(raw)
            if (raw not in assigned and isinstance(prior, int)
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
    for record in records:
        sid = record.get("speaker_id")
        if type(sid) is int and sid > 0:
            labeled.add(sid)
        if record.get("emb") is None:
            without_embedding += 1
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

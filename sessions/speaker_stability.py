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

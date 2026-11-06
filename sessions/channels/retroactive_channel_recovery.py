"""Conservative backfill after a Nemotron channel gains a stable identity.

Uses a confirmed channel with voice corroboration or strong local activity.
It never changes ASR text, timestamps, raw clusters or speaker embeddings.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping


def backfill_confirmed_channels(
    projected: list[dict],
    original: list[dict],
    channel_labels: Mapping[int, int],
    match_known: Callable[[object], int | None],
    *,
    minimum_activity: float = 0.75,
    minimum_margin: float = 0.40,
    effective_embeddings: Mapping[int, object] | None = None,
) -> dict[int, int]:
    if len(projected) != len(original):
        raise ValueError("Projected and original segment counts differ")
    changed = {}
    for item, source in zip(projected, original):
        label = item.get("speaker_id")
        if type(label) is not int or label > 0:
            continue
        if (source.get("speaker_override") or source.get("matched_profile_id")
                or item.get("speaker_diarization_source") != "nemotron3_uncertain"):
            continue
        if item.get("speaker_activity_status") not in (
                "official_turn_primary", "acoustic_primary"):
            continue
        try:
            coverage = float(item.get("speaker_activity_score"))
            margin = float(item.get("speaker_activity_margin"))
        except (ValueError, TypeError, OverflowError):
            continue
        if not coverage >= minimum_activity or not margin >= minimum_margin:
            continue
        channel = item.get("speaker_acoustic_channel")
        if type(channel) is not int or channel <= 0:
            continue
        stable = channel_labels.get(channel)
        if type(stable) is not int or stable <= 0:
            continue
        if source.get("speaker_evidence_status") == "multi_turn_acoustic_evidence_only":
            continue
        embedding = source.get("emb")
        if embedding is None and effective_embeddings is not None:
            embedding = effective_embeddings.get(int(item["id"]))
        if embedding is not None:
            if match_known(embedding) != stable:
                continue
        elif (coverage < 0.90 or margin < 0.65
                or item.get("speaker_channel_gate_reason") != "missing_voice_embedding"):
            # Only reconcile an earlier pending item on an already bound
            # channel. Never recover an explicit voice conflict or overlap.
            continue
        item["speaker_id"] = stable
        item["speaker_name"] = f"Speaker {stable}"
        item["speaker_diarization_source"] = "nemotron3_backfilled"
        item["speaker_status"] = "acoustic_backfilled_provisional"
        item["speaker_uncertain"] = True
        item["speaker_final_resolution_status"] = (
            "independent_acoustic_backfilled_provisional"
        )
        changed[int(item["id"])] = stable
    return changed


def reconcile_final_channels(
    items: list[dict], original: list[dict], official_turns: list[dict],
    channel_labels: Mapping[int, int], *, offset: int = 0,
) -> dict[int, int]:
    """Project sustained, exclusive finalized activity to provisional IDs.

    Only one label is allocated per supported unbound channel. Conflicting
    voiceprints on an already bound channel and human edits stay untouched.
    """
    events = []
    for turn in official_turns:
        channel, start, end = (turn.get(k) for k in ("speaker_id", "start_ms", "end_ms"))
        if (type(channel) is int and 1 <= channel <= 8
                and type(start) is int and type(end) is int and 0 <= start < end):
            events.extend(((start, channel, 1), (end, channel, -1)))
    exclusive_ms = {}
    active = {}
    previous = 0
    for time, channel, delta in sorted(events):
        if time > previous and len(active) == 1:
            sole = next(iter(active))
            exclusive_ms[sole] = exclusive_ms.get(sole, 0) + time - previous
        previous = time
        active[channel] = active.get(channel, 0) + delta
        if not active[channel]:
            del active[channel]

    groups = {}
    for item, source in zip(items, original, strict=True):
        channel = item.get("speaker_acoustic_channel")
        if (source.get("speaker_override") or source.get("matched_profile_id")
                or type(channel) is not int or channel in channel_labels
                or exclusive_ms.get(channel, 0) < 3000
                or item.get("speaker_activity_status") != "official_turn_primary"
                or item.get("speaker_diarization_source") not in (
                    "nemotron3_uncertain", "camplus_independent_unresolved",
                )):
            continue
        groups.setdefault(channel, []).append(item)

    # Require two separate, dominant caption intervals. Repeated ASR rows
    # at one timestamp do not establish another visible person.
    supported = []
    for group in groups.values():
        strong = sorted(
            (item for item in group if (item.get("speaker_activity_score") or 0) >= .60
             and (item.get("speaker_activity_margin") or 0) >= .25),
            key=lambda item: item["end_ms"],
        )
        if strong and any(item["start_ms"] >= strong[0]["end_ms"] for item in strong[1:]):
            supported.append(group)

    replaced_ids = {item["id"] for group in supported for item in group}
    used = set(channel_labels.values())
    used.update(item["speaker_id"] for item in items
                if item["id"] not in replaced_ids
                and type(item.get("speaker_id")) is int and item["speaker_id"] > 0)
    resolved = {}
    for group in supported:
        label = offset + 1
        while label in used:
            label += 1
        used.add(label)
        for item in group:
            item.update(
                speaker_id=label, speaker_name=f"Speaker {label}",
                speaker_diarization_source="nemotron3_channel_provisional",
                speaker_status="acoustic_channel_provisional",
                speaker_uncertain=True,
            )
            resolved[int(item["id"])] = label
    return resolved

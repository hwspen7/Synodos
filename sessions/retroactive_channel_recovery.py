"""Conservative backfill after a Nemotron channel gains a stable identity.

Only uses an already confirmed channel and an independently matching voice.
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
        embedding = source.get("emb")
        if embedding is None or match_known(embedding) != stable:
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

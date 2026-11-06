"""Assign a visible meeting-local speaker without promoting weak evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from speech.acoustic_timeline import normalize

PROVISIONAL_EXISTING = "existing_speaker_provisional"
PROVISIONAL_NEW = "new_speaker_provisional"
PROVISIONAL_RESOLUTIONS = frozenset({PROVISIONAL_EXISTING, PROVISIONAL_NEW})

_CHANNEL_SOURCES = frozenset({
    "nemotron3", "nemotron3_backfilled", "nemotron3_channel_provisional",
})
_CHANNEL_CONFLICTS = frozenset({
    "existing_channel_voice_conflict", "existing_channel_matches_other_voice",
    "candidate_voice_inconsistent", "candidate_voice_outlier",
})


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _channel(item: dict) -> int | None:
    channel = item.get("speaker_acoustic_channel")
    if (not _positive(channel) or item.get("speaker_channel_gate_reason") in _CHANNEL_CONFLICTS
            or item.get("speaker_activity_status") not in ("acoustic_primary", "official_turn_primary")
            or not (item.get("speaker_activity_score") or 0) >= 0.65
            or not (item.get("speaker_activity_margin") or 0) >= 0.25):
        return None
    return channel


def assign_provisional_speakers(
    items: list[dict],
    *,
    previous: Mapping[int, int] | None = None,
    channel_labels: Mapping[int, int] | None = None,
    voice_embeddings: Mapping[int, object] | None = None,
    reserved_labels: Iterable[int] = (),
) -> dict[int, int]:
    """Complete the display projection after all authoritative decisions.

    Reuse a uniquely supported channel or a brief, uninterrupted bridge
    between the same speaker. For everything else, allocate meeting-local
    placeholders. Placeholders never enter the channel mapper or voice
    clustering, and the original evidence fields are retained.
    """
    previous = previous or {}
    voice_embeddings = voice_embeddings or {}
    anchors: dict[int, set[int]] = defaultdict(set)
    for channel, label in (channel_labels or {}).items():
        if _positive(label):
            anchors[channel].add(label)
    used = {label for label in reserved_labels if _positive(label)}
    used.update(label for labels in anchors.values() for label in labels)

    ordered = sorted(items, key=lambda row: (row["start_ms"], row["id"]))
    established = {}
    names = {}
    voice_references = defaultdict(list)
    needs_voice_recovery = any(not _positive(item.get("speaker_id")) for item in ordered)
    for item in ordered:
        label = item.get("speaker_id")
        if not _positive(label):
            continue
        used.add(label)
        name = item.get("speaker_name")
        if not name or name == "Unidentified Speaker":
            item["speaker_name"] = f"Speaker {label}"
        if item.get("speaker_final_resolution_status") in PROVISIONAL_RESOLUTIONS:
            continue
        established[item["id"]] = label
        names[label] = item["speaker_name"]
        channel = _channel(item)
        if channel is not None and item.get("speaker_diarization_source") in _CHANNEL_SOURCES:
            anchors[channel].add(label)
        if (needs_voice_recovery and item["end_ms"] - item["start_ms"] >= 1200
                and not item.get("speaker_uncertain") and not item.get("overlap_intervals_ms")
                and item.get("speaker_evidence_status") in (
                    "turn_voiceprint", "acoustic_backfill", "speaker_birth_voice_reused",
                )):
            vector = normalize(voice_embeddings.get(item["id"], item.get("emb")))
            if vector is not None:
                references = voice_references[label]
                references.append((item["start_ms"], item["end_ms"], vector))
                if len(references) > 6:
                    del references[2]

    def apply(item: dict, label: int, reason: str) -> None:
        item.update(speaker_id=label, speaker_name=names.get(label, f"Speaker {label}"),
                    speaker_status="uncertain", speaker_uncertain=True,
                    speaker_final_resolution_status=reason)
        assignments[item["id"]] = label

    assignments = {}
    groups = defaultdict(list)
    last_run = None
    for index, item in enumerate(ordered):
        if _positive(item.get("speaker_id")):
            last_run = None
            continue
        channel = _channel(item)
        labels = anchors.get(channel, set())
        if len(labels) == 1:
            apply(item, next(iter(labels)), PROVISIONAL_EXISTING)
            last_run = None
            continue

        voice = voice_embeddings.get(item["id"], item.get("emb"))
        voice_eligible = (
            item.get("speaker_acoustic_channel") is None
            and not item.get("overlap_intervals_ms")
            and item.get("speaker_evidence_status") != "multi_turn_acoustic_evidence_only"
            and item.get("speaker_activity_status") not in (
                "acoustically_ambiguous", "official_turn_ambiguous",
            )
        )
        if (voice_references and voice_eligible
                and item["end_ms"] - item["start_ms"] >= 650):
            match = _match_voice(item, voice, voice_references)
            if match is not None:
                apply(item, match, PROVISIONAL_EXISTING)
                last_run = None
                continue

        left = ordered[index - 1] if index else None
        right = ordered[index + 1] if index + 1 < len(ordered) else None
        no_conflict = (
            voice_eligible and voice is None and not item.get("speaker_embedding_available")
        )
        if (no_conflict and left is not None and right is not None
                and established.get(left["id"]) == established.get(right["id"])
                and left["id"] in established
                and 0 <= item["start_ms"] - left["end_ms"] <= 450
                and 0 <= right["start_ms"] - item["end_ms"] <= 450
                and 0 < item["end_ms"] - item["start_ms"] <= 1200):
            apply(item, established[left["id"]], PROVISIONAL_EXISTING)
            last_run = None
            continue

        if channel is not None and not labels:
            key = ("channel", channel)
        elif (no_conflict and last_run is not None and left is not None
              and 0 <= item["start_ms"] - left["end_ms"] <= 450):
            key = last_run
        else:
            key = ("turn", item["id"])
        groups[key].append(item)
        last_run = key if no_conflict else None

    pending_ids = {item["id"] for group in groups.values() for item in group}
    reserved_previous = {
        label for sid, label in previous.items() if sid in pending_ids and _positive(label)
    }
    next_label = 1
    for group in groups.values():
        preferred = (previous.get(item["id"]) for item in group)
        label = next((value for value in preferred if _positive(value) and value not in used), None)
        if label is None:
            while next_label in used or next_label in reserved_previous:
                next_label += 1
            label = next_label
        used.add(label)
        for item in group:
            apply(item, label, PROVISIONAL_NEW)
    return assignments


def _match_voice(item: dict, raw: object, references: dict) -> int | None:
    probe = normalize(raw)
    if probe is None:
        return None
    ranked = sorted(
        ((max((float(np.dot(probe, vec)) for start, end, vec in turns
               if vec.shape == probe.shape
               and (end <= item["start_ms"] or start >= item["end_ms"])),
              default=-1.0), label)
         for label, turns in references.items()),
        reverse=True,
    )
    # Match the production strong-reuse cosine floor and require separation
    # from the runner-up. These are acoustic scores, not probabilities.
    score, label = ranked[0]
    if score >= 0.78 and (len(ranked) == 1 or score - ranked[1][0] >= 0.10):
        return label
    return None


def numbered_saved_segments(
    segments: list[dict], sample_rate: int, *, reserved_labels: Iterable[int] = (),
) -> list[dict]:
    """Project old saved transcripts without rewriting their stored records."""
    if all(_positive(item.get("speaker_id")) and item.get("speaker_name")
           and item["speaker_name"] != "Unidentified Speaker" for item in segments):
        return segments
    rows = [
        dict(item,
             start_ms=round(int(item["start_sample"]) * 1000 / sample_rate),
             end_ms=round(int(item["end_sample"]) * 1000 / sample_rate),
             speaker_embedding_available=item.get("emb") is not None)
        for item in segments
    ]
    assign_provisional_speakers(rows, reserved_labels=reserved_labels)
    fields = ("speaker_id", "speaker_name", "speaker_status",
              "speaker_uncertain", "speaker_final_resolution_status")
    return [
        {**source, **{key: row[key] for key in fields if key in row}}
        for source, row in zip(segments, rows, strict=True)
    ]

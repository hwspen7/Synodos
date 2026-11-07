"""Assign a visible meeting-local speaker without promoting weak evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from speech.audio.acoustic_timeline import normalize
from speech.supported_languages import PHASE1_LANGUAGE_SET

PROVISIONAL_EXISTING = "existing_speaker_provisional"
PROVISIONAL_NEW = "new_speaker_provisional"
ATTRIBUTION_DEFERRED = "speaker_attribution_deferred"
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


def _defer_attribution(item: dict) -> None:
    item.update(speaker_id=-1, speaker_name="", speaker_status="uncertain",
                speaker_uncertain=True, speaker_final_resolution_status=ATTRIBUTION_DEFERRED)


def _reliable_language(item: dict) -> str | None:
    language = item.get("language")
    confidence = item.get("language_confidence")
    return language if (isinstance(language, str) and language in PHASE1_LANGUAGE_SET
                        and isinstance(confidence, (int, float))
                        and 0.90 <= confidence <= 1.0) else None


def assign_provisional_speakers(
    items: list[dict],
    *,
    channel_labels: Mapping[int, int] | None = None,
    channel_stream_start: int = 0,
    voice_embeddings: Mapping[int, object] | None = None,
) -> dict[int, int]:
    """Complete the display projection after all authoritative decisions.

    Reuse a uniquely supported channel or a brief, uninterrupted bridge
    between the same speaker. Defer unassigned turns until additional
    evidence arrives. This projection cannot create speaker identities.
    """
    voice_embeddings = voice_embeddings or {}
    anchors: dict[tuple[int, int], set[int]] = defaultdict(set)
    for channel, label in (channel_labels or {}).items():
        if _positive(label):
            anchors[channel_stream_start, channel].add(label)
    for item in items:
        if item.get("speaker_final_resolution_status") == PROVISIONAL_NEW:
            _defer_attribution(item)
    ordered = sorted(items, key=lambda row: (row["start_ms"], row["id"]))
    established = {}
    names = {}
    voice_references = defaultdict(list)
    needs_voice_recovery = any(not _positive(item.get("speaker_id")) for item in ordered)
    for item in ordered:
        label = item.get("speaker_id")
        if not _positive(label):
            continue
        name = item.get("speaker_name")
        if not name or name == "Unidentified Speaker":
            item["speaker_name"] = f"Speaker {label}"
        if item.get("speaker_final_resolution_status") in PROVISIONAL_RESOLUTIONS:
            continue
        established[item["id"]] = label
        names[label] = item["speaker_name"]
        channel = _channel(item)
        if channel is not None and item.get("speaker_diarization_source") in _CHANNEL_SOURCES:
            anchors[item.get("speaker_acoustic_stream_start") or 0, channel].add(label)
        if (needs_voice_recovery and item["end_ms"] - item["start_ms"] >= 1200
                and not item.get("speaker_uncertain") and not item.get("overlap_intervals_ms")
                and item.get("speaker_evidence_status") in (
                    "turn_voiceprint", "acoustic_backfill", "speaker_birth_voice_reused",
                )):
            vector = normalize(voice_embeddings.get(item["id"], item.get("emb")))
            if vector is not None:
                references = voice_references[label]
                references.append((item["start_ms"], item["end_ms"], vector,
                                   _reliable_language(item)))
                if len(references) > 6:
                    del references[2]

    def apply(item: dict, label: int, reason: str) -> None:
        item.update(speaker_id=label, speaker_name=names.get(label, f"Speaker {label}"),
                    speaker_status="uncertain", speaker_uncertain=True,
                    speaker_final_resolution_status=reason)
        assignments[item["id"]] = label

    assignments = {}
    for index, item in enumerate(ordered):
        if _positive(item.get("speaker_id")):
            continue
        channel = _channel(item)
        labels = anchors.get((item.get("speaker_acoustic_stream_start") or 0, channel), set())
        if len(labels) == 1:
            apply(item, next(iter(labels)), PROVISIONAL_EXISTING)
            continue

        voice = voice_embeddings.get(item["id"], item.get("emb"))
        voice_eligible = (
            item.get("speaker_acoustic_channel") is None
            and item.get("speaker_channel_gate_reason") not in _CHANNEL_CONFLICTS
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
            continue

        _defer_attribution(item)
    return assignments


def _match_voice(item: dict, raw: object, references: dict) -> int | None:
    probe = normalize(raw)
    if probe is None:
        return None
    ranked = []
    for label, turns in references.items():
        eligible = [(float(np.dot(probe, vec)), start, end, language)
                    for start, end, vec, language in turns
                    if vec.shape == probe.shape
                    and (end <= item["start_ms"] or start >= item["end_ms"])]
        if eligible:
            ranked.append((max(row[0] for row in eligible), label, eligible))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if not ranked or ranked[0][0] < 0.78:
        return None
    score, label, eligible = ranked[0]
    if len(ranked) == 1 or score - ranked[1][0] >= 0.10:
        return label

    # Language can corroborate the acoustic winner in a narrow margin band.
    # Use only consistent, independently timed high-confidence references.
    language = _reliable_language(item)
    if language is None or score - ranked[1][0] < 0.06:
        return None
    if _reference_language(eligible) != language:
        return None
    for other_score, _, other in ranked[1:]:
        if score - other_score >= 0.10:
            break
        if _reference_language(other) in (None, language):
            return None
    return label


def _reference_language(eligible: list) -> str | None:
    languages = {row[3] for row in eligible if row[3] is not None}
    if len(languages) != 1:
        return None
    intervals = [(start, end) for _, start, end, language in eligible if language is not None]
    if min(end for _, end in intervals) <= max(start for start, _ in intervals):
        return next(iter(languages))
    return None


def project_saved_speaker_assignments(
    segments: list[dict], sample_rate: int,
) -> list[dict]:
    """Project old saved transcripts without rewriting their stored records."""
    if all(_positive(item.get("speaker_id")) and item.get("speaker_name")
           and item["speaker_name"] != "Unidentified Speaker"
           and item.get("speaker_final_resolution_status") not in
               (ATTRIBUTION_DEFERRED, PROVISIONAL_NEW)
           for item in segments):
        return segments
    rows = [
        dict(item,
             start_ms=round(int(item["start_sample"]) * 1000 / sample_rate),
             end_ms=round(int(item["end_sample"]) * 1000 / sample_rate),
             speaker_embedding_available=item.get("emb") is not None)
        for item in segments
    ]
    assign_provisional_speakers(rows)
    fields = ("speaker_id", "speaker_name", "speaker_status",
              "speaker_uncertain", "speaker_final_resolution_status")
    return [
        {**source, **{key: row[key] for key in fields if key in row}}
        for source, row in zip(segments, rows, strict=True)
    ]

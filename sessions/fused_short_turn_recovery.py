"""Conservative bidirectional recovery for unidentified short speaker turns.

This module runs *after* independent Nemotron channel corroboration.  It
never creates a speaker or changes a transcript, and never promotes a short
probe to a trusted reference. Unresolved speech remains unidentified.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


_FINAL_IDENTITY_SOURCES = frozenset({"nemotron3", "registered_voiceprint"})
_UNKNOWN_REASONS = frozenset({
    "short_probe", "short_probe_known_voice", "speaker_birth_unconfirmed",
    "speaker_birth_awaiting_voice", "turn_analysis_unavailable",
    "turn_split_rejected", "acoustic_backfill",
})


def _unit(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if vec.size < 2 or not np.isfinite(vec).all():
        return None
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-7:
        return None
    return vec / norm


def _interval_distance(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(b0 - a1, a0 - b1, 0)


def recover_fused_short_turns(
    projected: list[dict],
    original: list[dict],
    *,
    sample_rate: int,
    max_candidate_s: float = 1.35,
    min_reference_s: float = 1.0,
    max_reference_gap_s: float = 40.0,
    min_similarity: float = 0.88,
    short_similarity: float = 0.92,
    minimum_margin: float = 0.12,
    min_pair_agreement: float = 0.72,
) -> dict[int, int]:
    """Update projected, final-display labels only where evidence agrees.

    Require two distinct, time-separated, independently published reference
    turns of the *same* speaker. Their CAM++ embeddings must agree. Two
    reference similarities above the floor and a margin against every other
    eligible speaker are required. This supports future and past reference
    turns while excluding overlap and same-utterance fragment duplication.

    For sub-500-ms speech, demand an even higher floor. A present CAM++
    embedding is not proof of correctness: acoustic backfill can contain
    adjacent voice, especially for very short interjections.

    The original segment objects, transcript, timestamps, and embeddings
    are never modified. Return only the resolved segment-to-speaker map.
    """
    if (sample_rate <= 0 or len(projected) != len(original)
            or not 0.5 < min_similarity <= short_similarity <= 1.0
            or not 0.0 <= minimum_margin <= 0.5
            or not 0.0 < min_pair_agreement <= 1.0
            or not 0.0 < max_candidate_s <= 3.0
            or not 0.5 <= min_reference_s <= 5.0
            or not 1.0 <= max_reference_gap_s <= 120.0):
        raise ValueError("Invalid speaker recovery settings or record alignment")

    max_gap = round(max_reference_gap_s * sample_rate)
    min_ref = round(min_reference_s * sample_rate)
    max_candidate = round(max_candidate_s * sample_rate)
    groups: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)

    # Only independently corroborated published speaker IDs can become
    # references. In particular, CAM++ provisional IDs and any recovered
    # short turn are never eligible, preventing circular self-confirmation.
    for item, source in zip(projected, original):
        label = item.get("speaker_id")
        if (type(label) is not int or label <= 0
                or item.get("speaker_diarization_source") not in _FINAL_IDENTITY_SOURCES
                or source.get("speaker_override")
                or item.get("speaker_diarization_source") == "nemotron3"
                and item.get("speaker_activity_status") not in (
                    "acoustic_primary", "official_turn_primary")):
            continue
        start = int(source.get("start_sample") or 0)
        end = int(source.get("end_sample") or 0)
        if end - start < min_ref:
            continue
        voice = _unit(source.get("emb"))
        if voice is not None:
            groups[label].append((start, end, voice))

    reliable: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for label, turns in groups.items():
        # Sorted time intervals, not segment IDs, determine independence.
        turns.sort(key=lambda t: (t[0], t[1]))
        independent = []
        for start, end, vec in turns:
            if independent and start < independent[-1][1] + round(.25 * sample_rate):
                continue
            independent.append((start, end, vec))
        if len(independent) >= 2:
            reliable[label] = independent

    resolved: dict[int, int] = {}
    if not reliable:
        return resolved
    for item, source in zip(projected, original):
        if item.get("speaker_override") or source.get("speaker_override"):
            continue
        if item.get("matched_profile_id") or source.get("matched_profile_id"):
            continue
        if type(item.get("speaker_id")) is not int or item["speaker_id"] > 0:
            continue
        if item.get("speaker_diarization_source") not in (
                "nemotron3_uncertain", "camplus_provisional", "camplus"):
            continue
        start = int(source.get("start_sample") or 0)
        end = int(source.get("end_sample") or 0)
        duration = end - start
        if not 0 < duration <= max_candidate:
            continue
        reason = str(source.get("speaker_evidence_status") or "")
        upstream = str(source.get("speaker_upstream_evidence_status") or "")
        if reason not in _UNKNOWN_REASONS and upstream not in _UNKNOWN_REASONS:
            continue
        probe = _unit(source.get("emb"))
        if probe is None:
            continue

        # For each person, evaluate the closest two independent references.
        # Both must score well. Similarities are *not* probabilities.
        scores: list[tuple[float, int]] = []
        for label, turns in reliable.items():
            eligible = [
                (int(_interval_distance(start, end, left, right)), vec)
                for left, right, vec in turns
                if (right <= start or left >= end)
                and vec.shape == probe.shape
                and _interval_distance(start, end, left, right) <= max_gap
            ]
            if len(eligible) < 2:
                continue
            eligible.sort(key=lambda x: x[0])
            # Pick the closest two independent acoustic samples; limiting
            # distant references prevents a far-off voice from dominating.
            first, second = eligible[:2]
            if float(np.dot(first[1], second[1])) < min_pair_agreement:
                continue
            sims = (float(np.dot(probe, first[1])),
                    float(np.dot(probe, second[1])))
            scores.append((min(sims), label))
        scores.sort(key=lambda x: (-x[0], x[1]))
        floor = short_similarity if duration < round(.5 * sample_rate) else min_similarity
        if not scores or scores[0][0] < floor:
            continue
        if len(scores) > 1 and scores[0][0] - scores[1][0] < minimum_margin:
            continue
        label = scores[0][1]
        item["speaker_id"] = label
        item["speaker_name"] = f"Speaker {label}"
        item["speaker_status"] = "short_voice_recovered_provisional"
        item["speaker_uncertain"] = True
        item["speaker_diarization_source"] = "camplus_bidirectional_recovery"
        item["speaker_final_resolution_status"] = "voice_recovered_provisional"
        resolved[int(item["id"])] = label
    return resolved

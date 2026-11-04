"""Source-independent speaker activity scoring and caption alignment.

Speaker channels come from one continuous diarization session, not from
text/audio sentence boundaries. Channel 1 means the first arriving voice;
it is NOT a known human identity or a calibrated match probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

FRAME_MS = 10
MAX_SPEAKERS = 8


@dataclass(frozen=True)
class SpeakerDecision:
    speaker_id: int | None
    activity_score: float | None
    runner_up_score: float | None
    margin: float | None
    active_fraction: float | None
    analyzed_ms: int
    status: str

    def as_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "activity_score": self.activity_score,
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "active_fraction": self.active_fraction,
            "analyzed_ms": self.analyzed_ms,
            "status": self.status,
            "score_kind": "model_activity_not_identity_probability",
        }


def validate_activity(raw) -> np.ndarray:
    """Require model activity probabilities, never silently sigmoid twice."""
    data = np.asarray(raw, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != MAX_SPEAKERS:
        raise ValueError("Expected [T,8] Nemotron activity")
    if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
        raise ValueError("Nemotron output must be activity probabilities in [0,1]")
    return data


def choose_caption_speaker(
    activity, start_ms: int, end_ms: int, *,
    frame_ms: int = FRAME_MS,
    activity_threshold: float = 0.5,
    decision_threshold: float = 0.54,
    margin_threshold: float = 0.10,
    min_active_fraction: float = 0.25,
    min_coverage: float = 0.50,
) -> SpeakerDecision:
    """Assign a single *primary* caption channel, refusing ambiguous evidence.

    Keep all overlapping model channels on the separate acoustic timeline.
    Never use transcript text, reference RTTM, or a requested speaker count.
    The activity score is a temporal average, NOT a calibrated P(identity).
    """
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or end_ms <= start_ms
            or frame_ms <= 0
            or not all(math.isfinite(float(x)) and 0 <= x <= 1 for x in (
                activity_threshold, decision_threshold, margin_threshold,
                min_active_fraction, min_coverage,
            ))):
        raise ValueError("Invalid speaker alignment parameters")
    probabilities = validate_activity(activity)
    first = min(len(probabilities), start_ms // frame_ms)
    last = min(len(probabilities), (end_ms + frame_ms - 1) // frame_ms)
    expected = max(1, (end_ms - start_ms + frame_ms - 1) // frame_ms)
    count = max(0, last - first)
    if not count or count / expected < min_coverage:
        return SpeakerDecision(None, None, None, None, None,
                               count * frame_ms, "insufficient_timeline")
    sample = probabilities[first:last]
    means = sample.mean(axis=0)
    ranked = np.argsort(-means, kind="stable")
    top, second = int(ranked[0]), int(ranked[1])
    score = float(means[top])
    other = float(means[second])
    margin = score - other
    fraction = float(np.mean(sample[:, top] >= activity_threshold))
    if (score < decision_threshold or margin < margin_threshold
            or fraction < min_active_fraction):
        return SpeakerDecision(None, round(score, 5), round(other, 5),
                               round(margin, 5), round(fraction, 5),
                               count * frame_ms, "acoustically_ambiguous")
    return SpeakerDecision(top + 1, round(score, 5), round(other, 5),
                           round(margin, 5), round(fraction, 5),
                           count * frame_ms, "acoustic_primary")


def choose_caption_speaker_from_turns(
    turns, start_ms: int, end_ms: int, *,
    min_fraction: float = 0.25,
    min_margin: float = 0.10,
) -> SpeakerDecision:
    """Choose the dominant official diarization turn for one caption.

    Official turn IDs are the final source of truth. Unlike raw sigmoid
    channels this cannot introduce model channels suppressed by Nemotron's
    official speaker-turn postprocessing. Each speaker's intersecting
    intervals are unioned before measuring covered clock time.
    """
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or end_ms <= start_ms
            or not 0 <= min_fraction <= 1
            or not 0 <= min_margin <= 1):
        raise ValueError("Invalid official-turn caption projection")
    if not isinstance(turns, (list, tuple)):
        raise ValueError("Expected finalized Nemotron turns")
    covered: dict[int, list[tuple[int, int]]] = {}
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Invalid official Nemotron turn")
        speaker = turn.get("speaker_id")
        a = turn.get("start_ms")
        b = turn.get("end_ms")
        if (type(speaker) is not int or not 1 <= speaker <= MAX_SPEAKERS
                or type(a) is not int or type(b) is not int
                or a < 0 or b <= a):
            raise ValueError("Invalid official speaker interval")
        left, right = max(a, start_ms), min(b, end_ms)
        if right > left:
            covered.setdefault(speaker, []).append((left, right))
    if not covered:
        return SpeakerDecision(None, 0., 0., 0., 0.,
                               end_ms - start_ms, "official_turn_no_speech")

    shares = {}
    for speaker, fragments in covered.items():
        merged = []
        for a, b in sorted(fragments):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        shares[speaker] = sum(b - a for a, b in merged) / (end_ms - start_ms)
    ranked = sorted(shares.items(), key=lambda x: (-x[1], x[0]))
    top, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.
    margin = score - second
    if score < min_fraction or margin < min_margin:
        return SpeakerDecision(None, round(score, 5), round(second, 5),
                               round(margin, 5), round(score, 5),
                               end_ms - start_ms, "official_turn_ambiguous")
    return SpeakerDecision(top, round(score, 5), round(second, 5),
                           round(margin, 5), round(score, 5),
                           end_ms - start_ms, "official_turn_primary")


def activity_turns(
    activity, *, frame_ms: int = FRAME_MS,
    threshold: float = 0.5,
    min_turn_ms: int = 20,
) -> list[dict]:
    """Convert independent activity to parallel speaker intervals.

    Primarily a deterministic fallback/diagnostic; use the official
    processor.extract_speaker_dict where available for model-specific
    postprocessing. No channels are suppressed because another is active.
    """
    data = validate_activity(activity)
    if (frame_ms <= 0 or not 0 < threshold < 1
            or min_turn_ms < frame_ms):
        raise ValueError("Invalid activity interval configuration")
    turns = []
    for channel in range(data.shape[1]):
        active = data[:, channel] >= threshold
        onset = None
        for index in range(len(active) + 1):
            current = bool(active[index]) if index < len(active) else False
            if current and onset is None:
                onset = index
            elif not current and onset is not None:
                if (index - onset) * frame_ms >= min_turn_ms:
                    turns.append({
                        "start_ms": onset * frame_ms,
                        "end_ms": index * frame_ms,
                        "speaker_id": channel + 1,
                    })
                onset = None
    return sorted(turns, key=lambda row: (
        row["start_ms"], row["end_ms"], row["speaker_id"],
    ))

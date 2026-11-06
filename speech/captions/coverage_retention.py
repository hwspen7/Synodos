"""Bounded retention of acoustically detected but untranscribed speech.

Only VAD-backed intervals are considered. No speech text or identity is
fabricated. After limited repeated ASR opportunities a gap is explicitly
flagged for source-audio review rather than silently dropped.
"""
from __future__ import annotations

from speech.contracts import Utterance
from speech.captions.gap_recovery import uncovered_voice_intervals


def choose_safe_drop(
    *, proposed_drop_samples: int,
    audio_samples: int,
    sample_rate: int,
    global_start_samples: int,
    speech_intervals_ms: tuple[tuple[int, int], ...],
    utterances: tuple[Utterance, ...],
    attempts: dict[int, int],
    max_holds: int = 2,
    final: bool = False,
    min_uncovered_ms: int = 1000,
) -> tuple[int, list[dict]]:
    """Return a drop frontier and newly exhausted reference-free voice gaps.

    Attempts are per-session, keyed by absolute 250ms time buckets.
    Each gap can delay reclamation at most max_holds times. Every
    exhausted gap is a warning, not evidence of real words.
    """
    if (not 0 < sample_rate <= 192000 or audio_samples < 0
            or not 0 <= proposed_drop_samples <= audio_samples
            or global_start_samples < 0 or not 0 <= max_holds <= 5):
        raise ValueError("Invalid speech coverage reclamation geometry")
    if not speech_intervals_ms:
        return proposed_drop_samples, []
    duration_ms = audio_samples * 1000 // sample_rate
    regions = [
        {
            "start": int(a * sample_rate / 1000),
            "end": int(b * sample_rate / 1000),
        }
        for a, b in speech_intervals_ms
    ]
    gaps = uncovered_voice_intervals(
        regions, utterances, duration_ms, min_uncovered_ms=min_uncovered_ms,
    )
    safe_drop = proposed_drop_samples
    exhausted = []
    start_ms = global_start_samples * 1000 // sample_rate
    for a, b in gaps:
        if not final and a * sample_rate // 1000 >= safe_drop:
            continue
        key = (start_ms + a) // 250
        count = attempts.get(key, 0)
        if count < max_holds and not final:
            attempts[key] = count + 1
            safe_drop = min(safe_drop, max(0, (a - 180) * sample_rate // 1000))
            break
        exhausted.append({
            "start_ms": start_ms + a,
            "end_ms": start_ms + b,
            "status": "voice_detected_transcript_unavailable",
        })
        # Preserve the exhausted state so the same unresolved interval
        # cannot enter an infinite retry/drop cycle on later passes.
        attempts[key] = max_holds + 1
    return safe_drop, exhausted

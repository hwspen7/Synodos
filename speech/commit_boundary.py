"""Finalize a confirmed utterance at the end of its acoustic turn.

Whisper word timestamps and Qwen VAD-window timestamps are not interchangeable.
A *verified* speech-off transition can consume a small leftover tail after an
already committed ASR utterance, but only when the entire remainder is bounded,
followed by sufficient silence, and no later active region is crossed.
This helper does not infer or delete words.
"""
from __future__ import annotations

from collections.abc import Iterable


def completed_voice_drop(
    committed_samples: int,
    audio_samples: int,
    intervals_ms: Iterable[tuple[int, int]],
    *,
    sample_rate: int = 16000,
    max_tail_ms: int = 380,
    settle_ms: int = 500,
    pad_ms: int = 80,
) -> int:
    """Return a conservative buffer-local confirmed audio cursor."""
    if sample_rate <= 0 or audio_samples < 0 or committed_samples < 0:
        raise ValueError("Invalid audio cursor geometry")
    if not 0 <= committed_samples <= audio_samples:
        raise ValueError("Committed cursor lies outside audio")
    if not 0 <= max_tail_ms <= 1000 or settle_ms < 0 or pad_ms < 0:
        raise ValueError("Invalid acoustic boundary policy")
    duration_ms = audio_samples * 1000 // sample_rate
    committed_ms = committed_samples * 1000 // sample_rate
    spans = []
    for interval in intervals_ms:
        try:
            start, end = int(interval[0]), int(interval[1])
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
        if 0 <= start < end <= duration_ms:
            spans.append((start, end))
    spans.sort()
    for index, (start, end) in enumerate(spans):
        if not start <= committed_ms <= end:
            continue
        if not 0 < end - committed_ms <= max_tail_ms:
            continue
        if duration_ms - end < settle_ms:
            continue
        # Never cross into a separate voiced interval or another person's
        # near-immediate reply; a 120ms onset guard stays in the buffer.
        wanted = end + pad_ms
        if any(start2 < wanted + 120 for start2, _ in spans[index + 1:]):
            continue
        return min(audio_samples, wanted * sample_rate // 1000)
    return committed_samples


def completed_island_drop(
    committed_samples: int,
    audio_samples: int,
    intervals_ms: Iterable[tuple[int, int]],
    *,
    published_intervals_ms: Iterable[tuple[int, int]],
    held_intervals_ms: Iterable[tuple[int, int]] = (),
    sample_rate: int = 16000,
    max_suffix_ms: int = 850,
    minimum_coverage: float = 0.50,
    settle_ms: int = 600,
    pad_ms: int = 80,
) -> int:
    """Atomically retire an acoustic speech island covered by committed text.

    The text's word-end timestamp can underestimate the actual voice-off
    boundary. Consume only a *settled* VAD island for which this same cycle
    committed enough time-aligned content. Other VAD islands, withheld
    hypotheses, and substantial uncovered speech always remain in the buffer.
    This intentionally cannot resolve two different words uttered in the
    same uninterrupted voiced interval.
    """
    if sample_rate <= 0 or audio_samples < 0 or not 0 <= committed_samples <= audio_samples:
        raise ValueError("Invalid acoustic cursor")
    if not 0 < max_suffix_ms <= 1200 or not 0 < minimum_coverage <= 1:
        raise ValueError("Invalid island ownership policy")
    duration_ms = audio_samples * 1000 // sample_rate
    committed_ms = committed_samples * 1000 // sample_rate
    voiced = sorted(
        (int(left), int(right))
        for left, right in intervals_ms
        if 0 <= int(left) < int(right) <= duration_ms
    )
    published = sorted(
        (int(left), int(right))
        for left, right in published_intervals_ms
        if 0 <= int(left) < int(right) <= duration_ms
    )
    held = tuple(
        (int(left), int(right)) for left, right in held_intervals_ms
    )
    for index, (start, end) in enumerate(voiced):
        if not start <= committed_ms <= end:
            continue
        if not 0 < end - committed_ms <= max_suffix_ms:
            continue
        if duration_ms - end < settle_ms + pad_ms:
            continue
        # A new speech island begins too close to this one. Do not consume
        # the other participant's fast reply or a distinct short word.
        if any(next_start < end + pad_ms + 200
               for next_start, _ in voiced[index + 1:]):
            continue
        if any(max(start, a) < min(end + pad_ms, b) for a, b in held):
            continue
        covered = [
            (max(start, a), min(end, b))
            for a, b in published if a < end and b > start
        ]
        if not covered:
            continue
        if max(right for _, right in covered) < committed_ms - 45:
            continue
        # Text must begin close to voice onset. Otherwise this island may
        # contain preceding untranscribed speech that would be lost.
        if min(left for left, _ in covered) > start + 600:
            continue
        union_ms = 0
        cursor = start
        for left, right in sorted(covered):
            union_ms += max(0, right - max(left, cursor))
            cursor = max(cursor, right)
        if union_ms / (end - start) < minimum_coverage:
            continue
        return min(audio_samples, (end + pad_ms) * sample_rate // 1000)
    return committed_samples


def unclosed_voice_turn(
    start_ms: int, end_ms: int, audio_ms: int,
    intervals_ms: Iterable[tuple[int, int]], *, settle_ms: int = 600,
) -> bool:
    """Hold a candidate while its supporting VAD island is still open."""
    for left, right in intervals_ms:
        a, b = int(left), int(right)
        if a < end_ms and b > start_ms and b >= audio_ms - settle_ms:
            return True
    return False

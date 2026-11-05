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

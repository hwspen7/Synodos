"""Windowed automatic language detection for Whisper-style recognition.

Detected language labels are based on individual acoustic windows. The model
may still make incorrect language decisions, particularly for short speech.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Iterable
from speech.audio_windows import local_speech_regions, speech_windows
from speech.contracts import Recognition, Utterance, Word
from speech.quality_guard import verify_utterances


def _probability(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def recognize_windows(
    samples,
    speech_regions: Iterable[Mapping[str, int]],
    transcribe: Callable,
    kwargs: dict,
    *,
    sample_rate: int = 16000,
    unknown_threshold: float = 0.50,
    max_windows: int = 100,
    preserve_disputed_text: bool = False,
    short_voice_support_ms: int = 160,
) -> Recognition:
    """Call transcribe(chunk, **kwargs) separately for each VAD speech window.

    The callable returns (iterable_of_model_segments, info). Never silently
    truncate the number of audio windows or claim a global language label as
    a per-utterance language. Callers must serialize ASR model inference.
    """
    regions = [dict(x) for x in speech_regions]
    windows = speech_windows(regions, len(samples), sample_rate=sample_rate)
    if len(windows) > max_windows:
        raise RuntimeError("Too many speech windows; use a shorter ASR processing interval")
    pieces: list[Utterance] = []
    confirmed_languages: set[str] = set()
    for start, end in windows:
        local_regions = local_speech_regions(regions, start, end)
        if not local_regions:
            continue
        output, info = transcribe(samples[start:end], **kwargs)
        language = str(getattr(info, "language", "und") or "und").lower()
        probability = _probability(getattr(info, "language_probability", None))
        if probability is not None and probability < unknown_threshold:
            language = "und"
        if language not in ("auto", "", "und"):
            confirmed_languages.add(language)
        verified = verify_utterances(
            output, local_regions, round((end - start) * 1000 / sample_rate),
            language, preserve_disputed_text=preserve_disputed_text,
            short_voice_support_ms=short_voice_support_ms,
        )
        shift_ms = round(start * 1000 / sample_rate)
        for item in verified:
            words = tuple(Word(w.text, w.start_ms + shift_ms, w.end_ms + shift_ms) for w in item.words)
            pieces.append(Utterance(item.text, item.start_ms + shift_ms, item.end_ms + shift_ms,
                                    language, words, item.timestamp_quality, item.confidence,
                                    language_confidence=probability))
    pieces.sort(key=lambda item: (item.start_ms, item.end_ms))
    label = next(iter(confirmed_languages)) if len(confirmed_languages) == 1 else "mul" if confirmed_languages else "und"
    return Recognition(tuple(pieces), label, engine="whisper", speech_detected=bool(windows))

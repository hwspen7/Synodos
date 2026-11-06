"""Windowed automatic language detection for Whisper-style recognition.

Detected language labels are based on individual acoustic windows. The model
may still make incorrect language decisions, particularly for short speech.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Iterable
from speech.language import normalize_language
from speech.supported_languages import PHASE1_LANGUAGE_SET
from speech.audio.audio_windows import local_speech_regions, speech_windows
from speech.contracts import Recognition, Utterance, Word
from speech.captions.quality_guard import verify_utterances, speech_intervals
from speech.captions.asr_stage_trace import (
    merged_intervals, model_intervals, utterance_intervals,
)


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
    prefer_word_bounds: bool = False,
    allow_short_vad_supported: bool = False,
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
    trace_enabled = os.getenv("MEETING_ASR_TRACE_STAGES", "0").lower() in (
        "1", "true", "yes",
    )
    raw_model_intervals = []
    uncertain = []
    anchors = []
    for start, end in windows:
        local_regions = local_speech_regions(regions, start, end)
        if not local_regions:
            continue
        output, info = transcribe(samples[start:end], **kwargs)
        if trace_enabled:
            output = list(output)
            raw_model_intervals.extend(model_intervals(
                output, duration_ms=round(len(samples) * 1000 / sample_rate),
                shift_ms=round(start * 1000 / sample_rate),
            ))
        language = normalize_language(getattr(info, "language", None))
        probability = _probability(getattr(info, "language_probability", None))
        local_ms = round((end - start) * 1000 / sample_rate)
        verified = verify_utterances(
            output, local_regions, local_ms, language,
            preserve_disputed_text=preserve_disputed_text,
            prefer_word_bounds=prefer_word_bounds,
            allow_short_vad_supported=allow_short_vad_supported,
        )
        if not verified:
            continue
        span_ms = verified[-1].end_ms - verified[0].start_ms
        if (language in PHASE1_LANGUAGE_SET and probability is not None
                and probability >= .75 and span_ms >= 1500):
            anchors.append((start, end, language))
        uncertain.append((start, end, local_regions, verified, language, probability))

    retries = 0
    for start, end, local_regions, verified, language, probability in uncertain:
        nearby = sorted(
            (max(0, a - end, start - b), code) for a, b, code in anchors
            if max(0, a - end, start - b) <= 8 * sample_rate
        )
        context = nearby[0][1] if nearby and len({code for _, code in nearby}) == 1 else None
        span_ms = verified[-1].end_ms - verified[0].start_ms
        unsupported = language not in PHASE1_LANGUAGE_SET and language != "und"
        weak = probability is not None and probability < unknown_threshold
        abrupt_switch = (
            context is not None and context != language and span_ms < 1200
            and (probability is None or probability < .85)
        )
        if unsupported or (weak and span_ms < 1500) or abrupt_switch:
            if context is None or retries >= 2:
                continue
            retries += 1
            retry_options = dict(kwargs, language=context, multilingual=False)
            try:
                recovered, _ = transcribe(samples[start:end], **retry_options)
                verified = verify_utterances(
                    recovered, local_regions, round((end - start) * 1000 / sample_rate),
                    context, preserve_disputed_text=preserve_disputed_text,
                    prefer_word_bounds=prefer_word_bounds,
                    allow_short_vad_supported=allow_short_vad_supported,
                )
            except Exception:
                continue
            language, probability = context, None
        elif weak:
            language = "und"
        if verified and language != "und":
            confirmed_languages.add(language)
        shift_ms = round(start * 1000 / sample_rate)
        for item in verified:
            words = tuple(Word(w.text, w.start_ms + shift_ms, w.end_ms + shift_ms) for w in item.words)
            pieces.append(Utterance(item.text, item.start_ms + shift_ms, item.end_ms + shift_ms,
                                    language, words, item.timestamp_quality, item.confidence,
                                    language_confidence=probability))
    pieces.sort(key=lambda item: (item.start_ms, item.end_ms))
    label = next(iter(confirmed_languages)) if len(confirmed_languages) == 1 else "mul" if confirmed_languages else "und"
    trace = None
    if trace_enabled:
        duration_ms = round(len(samples) * 1000 / sample_rate)
        trace = {
            "vad_intervals_ms": [list(pair) for pair in speech_intervals(
                regions, duration_ms, sample_rate,
            )],
            "raw_model_intervals_ms": merged_intervals(
                raw_model_intervals, duration_ms=duration_ms,
            ),
            "verified_intervals_ms": utterance_intervals(
                pieces, duration_ms=duration_ms,
            ),
            "speech_windows": len(windows),
        }
    return Recognition(tuple(pieces), label, engine="whisper",
                       speech_detected=bool(windows),
                       recovery_diagnostics={"stage_trace": trace} if trace else None)

"""Bounded recovery of VAD-supported speech missing from ASR output.

Voice activity is acoustic evidence, not proof of intelligible words.
Recovery runs another *actual* ASR decode on uncovered audio. It does not
invent a transcript, bypass the VAD quality gate, or assign speakers.
"""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from speech.language import normalize_language
from speech.supported_languages import PHASE1_LANGUAGE_SET
from speech.contracts import Utterance, Word
from speech.quality_guard import speech_intervals, verify_utterances

MIN_UNCOVERED_MS = 900
EDGE_PAD_MS = 180
MAX_GAP_MS = 10000
MAX_RECOVERY_WINDOWS = 2


def uncovered_voice_intervals(
    vad_regions: Iterable[Mapping], utterances: Iterable[Utterance],
    duration_ms: int, *, min_uncovered_ms: int = MIN_UNCOVERED_MS,
    coverage_pad_ms: int = 160,
) -> tuple[tuple[int, int], ...]:
    """Return substantial VAD time without any verified transcript span.

    A short boundary gap is normal in forced alignment and is not
    treated as missing words. Speaker identities play no role here.
    """
    if duration_ms < 0 or not 200 <= min_uncovered_ms <= 10000:
        raise ValueError("Invalid coverage audit duration or threshold")
    voice = speech_intervals(vad_regions, duration_ms)
    aligned = []
    for item in utterances:
        start = max(0, int(item.start_ms) - coverage_pad_ms)
        end = min(duration_ms, int(item.end_ms) + coverage_pad_ms)
        if item.text.strip() and end > start:
            aligned.append((start, end))
    aligned.sort()
    result = []
    for left, right in voice:
        position = left
        for a, b in aligned:
            if b <= position:
                continue
            if a >= right:
                break
            if a > position and a - position >= min_uncovered_ms:
                result.append((position, min(a, right)))
            position = max(position, min(right, b))
            if position >= right:
                break
        if right - position >= min_uncovered_ms:
            result.append((position, right))
    return tuple(result)


def recover_missing_voice(
    samples: np.ndarray,
    speech_regions: Iterable[Mapping],
    primary: tuple[Utterance, ...],
    transcribe: Callable,
    kwargs: dict,
    *,
    language: str,
    sample_rate: int = 16000,
    max_windows: int = MAX_RECOVERY_WINDOWS,
    min_uncovered_ms: int = MIN_UNCOVERED_MS,
) -> tuple[tuple[Utterance, ...], dict]:
    """Retry only acoustically supported uncovered windows using the ASR.

    The callable is lazy-safe: its iterable is fully consumed here while
    the Whisper infer lock is held by the caller. Inference errors are
    reported as failed attempts and never erase previously verified text.
    """
    if sample_rate != 16000 or not 0 <= max_windows <= 4:
        raise ValueError("Unsupported recovery sample rate or budget")
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Require finite mono audio")
    duration_ms = round(len(audio) * 1000 / sample_rate)
    regions = tuple(dict(item) for item in speech_regions)
    # The research candidate can inspect shorter VAD-positive gaps.
    # It remains strictly capped by decode-count and full audio bounds.
    missing = uncovered_voice_intervals(
        regions, primary, duration_ms,
        min_uncovered_ms=min_uncovered_ms,
        coverage_pad_ms=80 if min_uncovered_ms < MIN_UNCOVERED_MS else 160,
    )
    if max_windows == 0 or not missing:
        return primary, {
            "uncovered_speech_intervals": missing,
            "recovery_attempted": 0,
            "recovery_confirmed": 0,
            "recovery_failed": 0,
        }
    found = list(primary)
    attempted = confirmed = failed = 0
    budget = max_windows
    for start, end in missing:
        if budget == 0:
            break
        # Limit the context and the number of extra decoding calls per
        # streaming ASR cycle. Longer VAD intervals stay reviewable.
        for a in range(start, end, MAX_GAP_MS):
            if budget == 0:
                break
            b = min(end, a + MAX_GAP_MS)
            if b - a < min_uncovered_ms:
                continue
            clip_a = max(0, a - EDGE_PAD_MS)
            clip_b = min(duration_ms, b + EDGE_PAD_MS)
            clip = np.ascontiguousarray(
                audio[clip_a * 16:clip_b * 16], dtype=np.float32,
            )
            local = []
            for item in regions:
                left = max(clip_a * 16, int(item["start"]))
                right = min(clip_b * 16, int(item["end"]))
                if right > left:
                    local.append({
                        "start": left - clip_a * 16,
                        "end": right - clip_a * 16,
                    })
            budget -= 1
            attempted += 1
            if not local:
                continue
            try:
                settings = dict(kwargs)
                settings["vad_filter"] = False
                chunks, info = transcribe(clip, **settings)
                detected = (
                    language if language not in ("auto", "und", "")
                    else str(getattr(info, "language", "und") or "und")
                )
                probability = getattr(info, "language_probability", None)
                if language in ("auto", "und", "") and (
                    normalize_language(detected) not in PHASE1_LANGUAGE_SET
                    or (isinstance(probability, (int, float)) and probability < 0.50)
                ):
                    continue
                candidates = verify_utterances(
                    list(chunks), local, clip_b - clip_a, detected,
                    preserve_disputed_text=True,
                    allow_short_vad_supported=(
                        min_uncovered_ms < MIN_UNCOVERED_MS
                    ),
                )
            except Exception:
                failed += 1
                continue
            for item in candidates:
                shifted = replace(
                    item,
                    start_ms=item.start_ms + clip_a,
                    end_ms=item.end_ms + clip_a,
                    words=tuple(
                        Word(w.text, w.start_ms + clip_a, w.end_ms + clip_a)
                        for w in item.words
                    ),
                    recognition_origin="gap_recovery",
                )
                # A second decode must not displace or duplicate an
                # already confirmed transcript at the same time.
                if any(
                    max(0, min(shifted.end_ms, old.end_ms)
                        - max(shifted.start_ms, old.start_ms))
                    >= 0.50 * (shifted.end_ms - shifted.start_ms)
                    for old in found
                ):
                    continue
                found.append(shifted)
                confirmed += 1
    found.sort(key=lambda item: (item.start_ms, item.end_ms))
    return tuple(found), {
        "uncovered_speech_intervals": missing,
        "recovery_attempted": attempted,
        "recovery_confirmed": confirmed,
        "recovery_failed": failed,
    }

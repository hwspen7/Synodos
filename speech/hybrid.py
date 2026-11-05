"""Opt-in language-routed Whisper/Qwen ASR with conservative fallback."""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from speech.contracts import Recognition
from speech.language import normalize_language
from speech.supported_languages import PHASE1_LANGUAGE_SET, spoken_language

WHISPER_LANGUAGES = frozenset(("en", "zh", "ar", "id"))
QWEN_LANGUAGES = frozenset(("fr", "es", "ja", "ru", "de", "hi"))
POLICY = "fleurs-validation-200-v1"
PAD_MS = 160
MAX_REROUTES = 24


def preferred_backend(language: str) -> str:
    code = normalize_language(language)
    if code in WHISPER_LANGUAGES:
        return "whisper"
    if code in QWEN_LANGUAGES:
        return "qwen3"
    raise ValueError("No approved hybrid route for language " + repr(language))


def has_text(result: Recognition) -> bool:
    return any(x.text.strip() for x in result.utterances)


def acoustic_activity(audio: np.ndarray) -> bool:
    """Broad fallback trigger, not a voice activity classifier."""
    return bool(audio.size >= 1600 and
                np.sqrt(np.mean(np.square(audio, dtype=np.float64))) >= .001)


def shift_utterances(items, offset_ms: int, bound_ms: int, language: str):
    shifted = []
    for item in items:
        a, b = item.start_ms + offset_ms, item.end_ms + offset_ms
        if not item.text.strip() or not 0 <= a < b <= bound_ms:
            continue
        words = tuple(
            replace(word, start_ms=word.start_ms + offset_ms,
                    end_ms=word.end_ms + offset_ms)
            for word in item.words
            if 0 <= word.start_ms + offset_ms <
               word.end_ms + offset_ms <= bound_ms
        )
        shifted.append(replace(item, start_ms=a, end_ms=b,
                               language=language, words=words))
    return tuple(shifted)


class HybridEngine:
    """Whisper auto-language baseline; Qwen enhancement for six languages.

    In auto mode Whisper is the language detector and transcript fallback.
    Qwen re-runs only bounded Whisper-labelled speech segments. This has
    measurable extra cost, so it remains opt-in until live-caption QA.
    """
    def __init__(self, whisper, qwen, *, full_language_window=False):
        self.whisper, self.qwen = whisper, qwen
        self.full_language_window = bool(full_language_window)

    def _get_model(self):
        self.whisper._get_model()
        self.qwen.prewarm()
        return self.whisper._get_model()

    def _call(self, name: str, audio, language, terms) -> Recognition:
        engine = self.whisper if name == "whisper" else self.qwen
        result = engine.recognize(audio, language=language, terms=terms)
        if not isinstance(result, Recognition):
            raise TypeError("Invalid ASR recognition contract")
        return result

    @staticmethod
    def _wrap(result: Recognition, diagnostics: dict) -> Recognition:
        return replace(result, engine="hybrid", recovery_diagnostics={
            **(result.recovery_diagnostics or {}),
            "hybrid": diagnostics,
        })

    def recognize(self, audio, language="auto", terms="") -> Recognition:
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Hybrid requires finite mono audio")
        if samples.size == 0:
            return Recognition(engine="hybrid", speech_detected=False)
        language = spoken_language(language)
        if language == "auto":
            return self._auto(samples, terms)
        return self._known(samples, language, terms)

    def _known(self, samples, language, terms):
        chosen = preferred_backend(language)
        other = "qwen3" if chosen == "whisper" else "whisper"
        audit = {"policy": POLICY, "mode": "known", "primary": chosen,
                 "fallback": other, "fallback_attempted": False,
                 "fallback_accepted": False}
        try:
            primary = self._call(chosen, samples, language, terms)
        except Exception as exc:
            audit["primary_error"] = type(exc).__name__
            primary = None
        if primary is None or (not has_text(primary) and
                               acoustic_activity(samples)):
            audit["fallback_attempted"] = True
            try:
                secondary = self._call(other, samples, language, terms)
                if has_text(secondary):
                    audit["fallback_accepted"] = True
                    return self._wrap(secondary, audit)
            except Exception as exc:
                audit["fallback_error"] = type(exc).__name__
        if primary is None:
            raise RuntimeError("No viable hybrid ASR result from either backend")
        return self._wrap(primary, audit)

    def _auto(self, samples, terms):
        audit = {"policy": POLICY, "mode": "whisper_window_language_detection",
                 "qwen_attempts": 0, "qwen_accepted": 0,
                 "qwen_failed": 0, "whisper_retained": 0,
                 "uncertain_language": 0}
        try:
            baseline = self._call("whisper", samples, "auto", terms)
        except Exception as exc:
            audit["whisper_error"] = type(exc).__name__
            try:
                recovered = self._call("qwen3", samples, "auto", terms)
                audit["whisper_failure_recovered"] = bool(has_text(recovered))
                if has_text(recovered):
                    return self._wrap(recovered, audit)
            except Exception as second:
                audit["qwen_error"] = type(second).__name__
            raise RuntimeError("Both automatic-language ASR paths failed") from exc
        if not baseline.utterances:
            if acoustic_activity(samples):
                try:
                    qwen = self._call("qwen3", samples, "auto", terms)
                    if has_text(qwen):
                        audit["fallback_accepted"] = True
                        return self._wrap(qwen, audit)
                except Exception as exc:
                    audit["fallback_error"] = type(exc).__name__
            return self._wrap(baseline, audit)

        if self.full_language_window:
            selected_languages = {
                normalize_language(item.language) for item in baseline.utterances
            }
            reliable = all(
                item.language_confidence is None or (
                    isinstance(item.language_confidence, (float, int))
                    and math.isfinite(item.language_confidence)
                    and item.language_confidence >= .60
                ) for item in baseline.utterances
            )
            # An unmodified bounded recording avoids text-derived cuts
            # and the extra VAD boundary. This route is intentionally
            # restricted to a homogeneous short acoustic interval.
            if (samples.size <= 30 * 16000 and reliable
                    and len(selected_languages) == 1
                    and next(iter(selected_languages)) in QWEN_LANGUAGES):
                audit["strategy"] = "full_single_language_v2"
                audit["qwen_attempts"] += 1
                try:
                    # Preserve the standalone Qwen auto-language inference
                    # configuration: model choice is determined by Whisper
                    # language ID, but Qwen is not forced to use that label.
                    candidate = self._call(
                        "qwen3", samples, "auto", terms
                    )
                    if has_text(candidate):
                        audit["qwen_accepted"] += 1
                        return self._wrap(candidate, audit)
                    audit["qwen_empty"] = 1
                except Exception:
                    audit["qwen_failed"] += 1
                audit["whisper_retained"] = len(baseline.utterances)
                return self._wrap(baseline, audit)

        duration_ms = round(samples.size / 16)
        output = []
        for item in baseline.utterances:
            language = normalize_language(item.language)
            score = item.language_confidence
            reliable = (score is None or (
                isinstance(score, (float, int)) and
                math.isfinite(score) and score >= .60))
            if not reliable or language not in PHASE1_LANGUAGE_SET:
                audit["uncertain_language"] += 1
            if (not reliable or language not in QWEN_LANGUAGES or
                    audit["qwen_attempts"] >= MAX_REROUTES):
                output.append(item)
                audit["whisper_retained"] += 1
                continue
            start_ms = max(0, item.start_ms - PAD_MS)
            end_ms = min(duration_ms, item.end_ms + PAD_MS)
            start = start_ms * 16
            end = min(samples.size, end_ms * 16)
            if end - start < 1600:
                output.append(item)
                audit["whisper_retained"] += 1
                continue
            audit["qwen_attempts"] += 1
            try:
                qwen = self._call("qwen3", samples[start:end], language, terms)
                moved = shift_utterances(qwen.utterances, start_ms,
                                         duration_ms, language)
                if moved:
                    output.extend(moved)
                    audit["qwen_accepted"] += 1
                    continue
            except Exception:
                audit["qwen_failed"] += 1
            output.append(item)
            audit["whisper_retained"] += 1
        output.sort(key=lambda x: (x.start_ms, x.end_ms))
        result = replace(baseline, utterances=tuple(output))
        return self._wrap(result, audit)

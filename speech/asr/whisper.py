"""Faster Whisper engine; multisegment, multilingual and word timestamp capable."""
from __future__ import annotations
import threading
import os
import numpy as np
from speech.audio.dual_vad import crosscheck_speech_regions
from speech.contracts import Recognition
from speech.captions.vocabulary import prompt_terms
from speech.captions.quality_guard import verify_utterances, speech_intervals
from speech.captions.gap_recovery import recover_missing_voice
from speech.captions.utterance_units import split_utterances
from speech.captions.quality_profile import quality_candidate_enabled
from speech.captions.asr_stage_trace import model_intervals, utterance_intervals


class WhisperEngine:
    def __init__(self, model: str, device: str = "auto", compute_type: str = "default"):
        self.model_id, self.device, self.compute_type = model, device, compute_type
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _get_model(self):
        with self._load_lock:
            if self._model is None:
                if self.device != "cpu":
                    from speech.cuda_runtime import prepare_windows_cuda_runtime
                    prepare_windows_cuda_runtime()
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_id, device=self.device, compute_type=self.compute_type)
            return self._model

    def recognize(self, audio: np.ndarray, language: str = "auto", terms: str = "") -> Recognition:
        if len(audio) == 0:
            return Recognition(engine="whisper")
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        samples = np.asarray(audio, dtype=np.float32)
        options = VadOptions(
            min_speech_duration_ms=200,
            min_silence_duration_ms=650,
            speech_pad_ms=180,
        )
        # Preserve the exact V6 decoding path by default. Skipping the
        # model's internal VAD is an independent, experimental A/B variable.
        # A different VAD crop changes decoding context and can increase
        # both missed speech and hallucinations.
        # The high-level quality candidate applies to explicit languages,
        # too. Otherwise forced-language calls retained the second VAD
        # even when windowed automatic-language calls bypassed it.
        quality_candidate = quality_candidate_enabled()
        bypass_internal_vad = quality_candidate or os.environ.get(
            "MEETING_WHISPER_SKIP_INTERNAL_VAD", "0"
        ).strip().lower() in ("1", "true", "yes")
        kwargs = {
            "language": None if language in ("auto", "und", "") else language,
            "multilingual": language in ("auto", "und", ""),
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": not bypass_internal_vad,
            "vad_parameters": {
                "min_speech_duration_ms": options.min_speech_duration_ms,
                "min_silence_duration_ms": options.min_silence_duration_ms,
                "speech_pad_ms": options.speech_pad_ms,
            },
            "no_speech_threshold": 0.5,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "hallucination_silence_threshold": 1.2,
            "condition_on_previous_text": False,
            "temperature": 0,
        }
        hints = prompt_terms(terms.splitlines())
        if hints:
            kwargs["hotwords"] = ", ".join(hints)
        # Preserve acoustically verified segment words when the model's
        # forced word alignment is incomplete. This avoids deleting genuine
        # meeting content merely because word timestamps disagree.
        prefer_word_bounds = os.getenv(
            "MEETING_CAPTION_WORD_BOUNDS", "0"
        ).strip().lower() in ("1", "true", "yes")
        preserve_disputed = quality_candidate or os.environ.get(
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT", "0"
        ).strip().lower() in ("1", "true", "yes")
        gap_recovery = quality_candidate or os.environ.get(
            "MEETING_WHISPER_GAP_RECOVERY", "0"
        ).strip().lower() in ("1", "true", "yes")
        trace_enabled = os.getenv("MEETING_ASR_TRACE_STAGES", "0").lower() in (
            "1", "true", "yes",
        )
        stage_trace = None
        with self._infer_lock:
            speech_regions = get_speech_timestamps(
                samples, options, sampling_rate=16000
            )
            if os.environ.get(
                "MEETING_WHISPER_DUAL_VAD", "0"
            ).strip().lower() in ("1", "true", "yes"):
                speech_regions = crosscheck_speech_regions(samples, speech_regions)
            if not speech_regions:
                if trace_enabled:
                    stage_trace = {
                        "vad_intervals_ms": [],
                        "raw_model_intervals_ms": [],
                        "verified_intervals_ms": [],
                        "speech_windows": 0,
                    }
                return Recognition(
                    engine="whisper", speech_detected=False,
                    recovery_diagnostics=(
                        {"stage_trace": stage_trace} if stage_trace else None
                    ),
                )
            intervals_ms = speech_intervals(
                speech_regions, round(len(samples) * 1000 / 16000)
            )

            window_lid = (
                language in ("auto", "und", "")
                and os.getenv("MEETING_WHISPER_WINDOW_LID", "1").strip().lower()
                in ("1", "true", "yes")
            )
            if window_lid:
                from speech.asr.multilingual_windows import recognize_windows
                # The outer Silero VAD already bounds each speech window.
                # Under the paired quality candidate, do not apply a second
                # VAD crop inside each short Whisper decoder window.
                window_kwargs = dict(kwargs, multilingual=True)
                # Silero already supplied bounded speech windows. Running
                # Faster-Whisper VAD again inside each short window may
                # remove quiet onsets, but bypassing it is an A/B candidate,
                # not an assumed accuracy improvement.
                window_vad_bypass = os.getenv(
                    "MEETING_WHISPER_WINDOW_VAD_BYPASS", "0"
                ).strip().lower() in ("1", "true", "yes")
                if quality_candidate or window_vad_bypass:
                    window_kwargs["vad_filter"] = False
                recognized = recognize_windows(
                    samples, speech_regions,
                    lambda chunk, **kw: self._get_model().transcribe(chunk, **kw),
                    window_kwargs,
                    preserve_disputed_text=preserve_disputed,
                    prefer_word_bounds=prefer_word_bounds,
                    allow_short_vad_supported=quality_candidate,
                )
                primary = recognized.utterances
                detected = recognized.language
                probability = recognized.confidence
                if trace_enabled:
                    stage_trace = (
                        recognized.recovery_diagnostics or {}
                    ).get("stage_trace")
            else:
                # The iterator is lazy, and must be exhausted under the lock.
                segments, info = self._get_model().transcribe(samples, **kwargs)
                pieces = list(segments)
                detected = (
                    language if language not in ("auto", "und", "")
                    else getattr(info, "language", None) or "und"
                )
                probability = getattr(info, "language_probability", None)
                primary = verify_utterances(
                    pieces, speech_regions,
                    round(len(samples) * 1000 / 16000), detected,
                    preserve_disputed_text=preserve_disputed,
                    prefer_word_bounds=prefer_word_bounds,
                    allow_short_vad_supported=quality_candidate,
                )
                if trace_enabled:
                    duration_ms = round(len(samples) * 1000 / 16000)
                    stage_trace = {
                        "vad_intervals_ms": [list(pair) for pair in intervals_ms],
                        "raw_model_intervals_ms": model_intervals(
                            pieces, duration_ms=duration_ms,
                        ),
                        "verified_intervals_ms": utterance_intervals(
                            primary, duration_ms=duration_ms,
                        ),
                        "speech_windows": 1,
                    }

            # Bounded actual ASR retry for a VAD-positive interval omitted
            # from accepted speech. No text is fabricated, and the original
            # transcription is retained if retry fails.
            if gap_recovery:
                primary, recovery = recover_missing_voice(
                    samples, speech_regions, primary,
                    lambda chunk, **kw: self._get_model().transcribe(chunk, **kw),
                    kwargs, language=language,
                    max_windows=3 if quality_candidate else 2,
                    min_uncovered_ms=350 if quality_candidate else 900,
                    preserve_disputed_text=preserve_disputed,
                )
            else:
                recovery = {
                    "uncovered_speech_intervals": (),
                    "recovery_attempted": 0,
                    "recovery_confirmed": 0,
                    "recovery_failed": 0,
                    "disabled": True,
                }
        if trace_enabled and stage_trace is not None:
            stage_trace["window_internal_vad_bypassed"] = bool(
                window_lid and (
                    quality_candidate or os.getenv(
                        "MEETING_WHISPER_WINDOW_VAD_BYPASS", "0"
                    ).strip().lower() in ("1", "true", "yes")
                )
            )
            stage_trace["recovery_attempted"] = recovery.get(
                "recovery_attempted", 0,
            )
            stage_trace["recovery_confirmed"] = recovery.get(
                "recovery_confirmed", 0,
            )
            stage_trace["recovery_failed"] = recovery.get(
                "recovery_failed", 0,
            )
            stage_trace["after_recovery_intervals_ms"] = utterance_intervals(
                primary, duration_ms=round(len(samples) * 1000 / 16000),
            )
            recovery["stage_trace"] = stage_trace
        return Recognition(
            utterances=split_utterances(primary),
            language=detected,
            confidence=probability,
            engine="whisper",
            speech_detected=True,
            speech_intervals_ms=intervals_ms,
            recovery_diagnostics=recovery,
        )

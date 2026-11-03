"""Faster Whisper engine; multisegment, multilingual and word timestamp capable."""
from __future__ import annotations
import threading
import os
import numpy as np
from speech.dual_vad import crosscheck_speech_regions
from speech.contracts import Recognition
from speech.quality_guard import verify_utterances
from speech.utterance_units import split_utterances


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
        # Use the same VAD options for inference and output verification.
        # Reject only text with no supporting speech, never based on wording.
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        samples = np.asarray(audio, dtype=np.float32)
        options = VadOptions(min_speech_duration_ms=200, min_silence_duration_ms=650,
                             speech_pad_ms=180)
        kwargs = {
            "language": None if language in ("auto", "und", "") else language,
            "multilingual": language in ("auto", "und", ""),
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
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
        if terms.strip():
            kwargs["hotwords"] = terms[:1500]
        # Controlled experiment only: preserve verified segment text when
        # its forced word alignment fails acoustic support checks.
        # Disabled by default until all-language and overlap regression gates.
        preserve_disputed = os.environ.get(
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT", "0"
        ).strip().lower() in ("1", "true", "yes")
        with self._infer_lock:
            speech_regions = get_speech_timestamps(samples, options, sampling_rate=16000)
            if os.environ.get("MEETING_WHISPER_DUAL_VAD", "0").strip().lower() in ("1", "true", "yes"):
                speech_regions = crosscheck_speech_regions(samples, speech_regions)
            if not speech_regions:
                return Recognition(engine="whisper", speech_detected=False)
            # Auto LID is per VAD speech window, never a global label
            # copied onto every utterance. Keep an escape hatch for A/B tests.
            if language in ("auto", "und", "") and os.getenv(
                "MEETING_WHISPER_WINDOW_LID", "1"
            ).strip().lower() in ("1", "true", "yes"):
                from speech.multilingual_windows import recognize_windows
                window_kwargs = dict(kwargs, multilingual=True)
                from dataclasses import replace as dataclass_replace
                recognized = recognize_windows(
                    samples, speech_regions,
                    lambda chunk, **kw: self._get_model().transcribe(chunk, **kw),
                    window_kwargs, preserve_disputed_text=preserve_disputed,
                )
                return dataclass_replace(
                    recognized,
                    utterances=split_utterances(recognized.utterances),
                )
            segments, info = self._get_model().transcribe(samples, **kwargs)
            # Faster Whisper infers lazily; consume under the inference lock.
            pieces = list(segments)
        detected = (language if language not in ("auto", "und", "")
                    else getattr(info, "language", None) or "und")
        utterances = verify_utterances(
            pieces, speech_regions, round(len(samples) * 1000 / 16000), detected,
            preserve_disputed_text=preserve_disputed,
        )
        return Recognition(split_utterances(utterances), detected,
                           getattr(info, "language_probability", None), "whisper",
                           speech_detected=True)

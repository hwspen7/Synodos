"""Qwen3 ASR offline adapter with optional forced alignment."""
from __future__ import annotations
import threading
import numpy as np
from speech.contracts import Recognition, Utterance, Word
from speech.language import normalize_language

LANGUAGES = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French",
             "de": "German", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
             "pt": "Portuguese", "ru": "Russian", "it": "Italian"}


class QwenEngine:
    def __init__(self, model: str, device: str = "cuda:0", aligner: bool = True):
        self.model_id = model if model.startswith("Qwen/") else f"Qwen/{model}"
        self.device, self.use_aligner = device, aligner
        self._model = None
        self._lock = threading.RLock()

    def _get_model(self):
        if self._model is None:
            import torch
            from qwen_asr import Qwen3ASRModel
            kw = {"dtype": torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                  "device_map": self.device}
            if self.use_aligner:
                kw["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
                kw["forced_aligner_kwargs"] = {"dtype": kw["dtype"], "device_map": self.device}
            self._model = Qwen3ASRModel.from_pretrained(self.model_id, **kw)
        return self._model

    def recognize(self, audio: np.ndarray, language: str = "auto", terms: str = "") -> Recognition:
        if len(audio) == 0:
            return Recognition(engine="qwen3", speech_detected=False)
        import warnings
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        from speech.audio_windows import speech_windows
        from speech.dual_vad import crosscheck_speech_regions
        import os
        samples = np.asarray(audio, dtype=np.float32)
        # Independent acoustic speech boundaries prevent a no-alignment
        # transcript from spanning the entire rolling buffer indefinitely.
        vad = VadOptions(min_speech_duration_ms=200, min_silence_duration_ms=600, speech_pad_ms=150)
        regions = get_speech_timestamps(samples, vad, sampling_rate=16000)
        if os.getenv("MEETING_WHISPER_DUAL_VAD", "0").lower() in ("1", "true", "yes"):
            regions = crosscheck_speech_regions(samples, regions)
        if not regions:
            return Recognition(engine="qwen3", speech_detected=False)
        windows = speech_windows(regions, len(samples), max_window_s=18)
        if len(windows) > 100:
            raise RuntimeError("Too many ASR windows; process a shorter interval")
        if terms.strip():
            warnings.warn("Qwen3-ASR has no vocabulary prompt support in this adapter", RuntimeWarning)
        utterances = []
        recognized_languages = set()
        with self._lock:
            model = self._get_model()
            for start, end in windows:
                window = samples[start:end]
                arguments = {"audio": (window, 16000),
                             "language": None if language in ("auto", "und", "") else LANGUAGES.get(language, language)}
                try:
                    result = model.transcribe(**arguments, return_time_stamps=self.use_aligner)[0]
                except (ValueError, RuntimeError) as exc:
                    if not self.use_aligner:
                        raise
                    warnings.warn(f"Forced alignment failed; preserving text without word timing: {exc}", RuntimeWarning)
                    result = model.transcribe(**arguments, return_time_stamps=False)[0]
                detected = normalize_language(str(getattr(result, "language", language) or "und"))
                text = str(getattr(result, "text", "") or "").strip()
                if not text:
                    continue
                if detected not in ("und", "auto", ""):
                    recognized_languages.add(detected)
                units = []
                for stamp in getattr(result, "time_stamps", None) or []:
                    value = getattr(stamp, "text", None) or getattr(stamp, "word", None) or ""
                    a, b = getattr(stamp, "start_time", None), getattr(stamp, "end_time", None)
                    if a is None or b is None:
                        continue
                    a_ms, b_ms = round(float(a) * 1000), round(float(b) * 1000)
                    if 0 <= a_ms < b_ms <= round(len(window) / 16) + 250:
                        units.append(Word(str(value), round(start / 16) + a_ms, round(start / 16) + b_ms))
                if units:
                    begin, finish, quality = units[0].start_ms, units[-1].end_ms, "word"
                else:
                    # Window is VAD-bounded, not a manufactured word timestamp.
                    begin, finish, quality = round(start / 16), round(end / 16), "window_estimate"
                if finish > begin:
                    utterances.append(Utterance(text, begin, finish, detected, tuple(units), quality))
        label = next(iter(recognized_languages)) if len(recognized_languages) == 1 else "mul" if recognized_languages else "und"
        return Recognition(tuple(utterances), label, engine="qwen3", speech_detected=True)

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
            return Recognition(engine="qwen3")
        import warnings
        if terms.strip():
            warnings.warn("Qwen3-ASR adapter does not support vocabulary prompts; the transcript may omit configured hotwords", RuntimeWarning)
        with self._lock:
            model = self._get_model()
            arguments = {"audio": (np.asarray(audio, dtype=np.float32), 16000),
                         "language": None if language == "auto" else LANGUAGES.get(language, language)}
            try:
                result = model.transcribe(**arguments, return_time_stamps=self.use_aligner)[0]
            except (ValueError, RuntimeError) as exc:
                if not self.use_aligner:
                    raise
                warnings.warn(f"Qwen3 forced alignment unavailable; preserving text without word timings: {exc}", RuntimeWarning)
                result = model.transcribe(**arguments, return_time_stamps=False)[0]
        detected = normalize_language(str(getattr(result, "language", language) or "und"))
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return Recognition(language=detected, engine="qwen3")
        timestamps = getattr(result, "time_stamps", None) or []
        units = []
        if timestamps:
            for stamp in timestamps:
                value = getattr(stamp, "text", None) or getattr(stamp, "word", None) or ""
                start = getattr(stamp, "start_time", None)
                end = getattr(stamp, "end_time", None)
                if start is not None and end is not None:
                    units.append(Word(str(value), round(float(start) * 1000), round(float(end) * 1000)))
        if units:
            begin, end, quality = units[0].start_ms, units[-1].end_ms, "word"
        else:
            # Model output has no alignment. A window boundary is not a true word timestamp.
            begin, end, quality = 0, round(len(audio) / 16), "window_estimate"
        return Recognition((Utterance(text, begin, end, detected, tuple(units), quality),),
                           detected, engine="qwen3")

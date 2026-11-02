"""Lazy per-process engine registry preserving the original ASR fallback."""
from __future__ import annotations
import threading
import numpy as np
from speech.config import SpeechSettings, load_settings
from speech.contracts import Recognition, Utterance

_LOCK = threading.RLock()
_ENGINES = {}


def recognize(audio: np.ndarray, *, language: str = "auto", terms: str = "",
              config: SpeechSettings | None = None, legacy=None) -> Recognition:
    config = config or load_settings()
    if config.engine == "legacy":
        if legacy is None:
            raise RuntimeError("Legacy ASR callable is required")
        rows = legacy(audio, hotword=terms or None)
        return Recognition(tuple(Utterance(str(r.get("text", "")), int(r["start"]), int(r["end"]),
                                          "zh", timestamp_quality="segment") for r in rows), "zh", engine="legacy")
    from speech.enhancement import optional_denoise
    filtered = optional_denoise(audio, enabled=config.enable_denoise)
    key = (config.engine, config.model, config.device, config.compute_type)
    with _LOCK:
        if key not in _ENGINES:
            if config.engine == "whisper":
                from speech.whisper import WhisperEngine
                _ENGINES[key] = WhisperEngine(config.model, config.device, config.compute_type)
            else:
                from speech.qwen import QwenEngine
                _ENGINES[key] = QwenEngine(config.model, config.device if config.device != "auto" else "cuda:0")
        backend = _ENGINES[key]
    return backend.recognize(filtered, language, terms)

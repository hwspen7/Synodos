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
    key = (config.engine, config.model, config.device, config.compute_type,
           config.hybrid_qwen_python, config.hybrid_qwen_model)
    with _LOCK:
        if key not in _ENGINES:
            if config.engine == "whisper":
                from speech.whisper import WhisperEngine
                _ENGINES[key] = WhisperEngine(config.model, config.device, config.compute_type)
            elif config.engine == "hybrid":
                from speech.hybrid import HybridEngine
                from speech.qwen_bridge import QwenWorkerClient
                from speech.whisper import WhisperEngine
                _ENGINES[key] = HybridEngine(
                    WhisperEngine(config.model, config.device, config.compute_type),
                    QwenWorkerClient(config.hybrid_qwen_python,
                                     config.hybrid_qwen_model),
                    full_language_window=True,
                )
            else:
                from speech.qwen import QwenEngine
                import os
                aligner = os.getenv("MEETING_QWEN_FORCED_ALIGNER", "1").lower() not in ("0", "false", "no")
                _ENGINES[key] = QwenEngine(config.model, config.device if config.device != "auto" else "cuda:0", aligner=aligner)
        backend = _ENGINES[key]
    return backend.recognize(filtered, language, terms)



def prewarm_multilingual(config: SpeechSettings | None = None) -> None:
    """Load the configured ASR model before accepting a live recording.

    Cold model loading is intentionally visible at startup. It must not happen
    on the microphone ingestion path without the operator being aware.
    """
    config = config or load_settings()
    if config.engine == "legacy":
        return
    key = (config.engine, config.model, config.device, config.compute_type,
           config.hybrid_qwen_python, config.hybrid_qwen_model)
    with _LOCK:
        if key not in _ENGINES:
            if config.engine == "whisper":
                from speech.whisper import WhisperEngine
                _ENGINES[key] = WhisperEngine(config.model, config.device, config.compute_type)
            elif config.engine == "hybrid":
                from speech.hybrid import HybridEngine
                from speech.qwen_bridge import QwenWorkerClient
                from speech.whisper import WhisperEngine
                _ENGINES[key] = HybridEngine(
                    WhisperEngine(config.model, config.device, config.compute_type),
                    QwenWorkerClient(config.hybrid_qwen_python,
                                     config.hybrid_qwen_model),
                    full_language_window=True,
                )
            else:
                from speech.qwen import QwenEngine
                import os
                aligner = os.getenv("MEETING_QWEN_FORCED_ALIGNER", "1").lower() not in ("0", "false", "no")
                _ENGINES[key] = QwenEngine(config.model, config.device if config.device != "auto" else "cuda:0", aligner=aligner)
        backend = _ENGINES[key]
    backend._get_model()
    if config.engine in ("whisper", "hybrid") and config.device.lower().startswith("cuda"):
        # GPU dependencies are often loaded lazily on the first actual
        # transcription. Fail during startup rather than appearing healthy
        # and silently producing no text during a live recording.
        probe = np.zeros(3 * 16000, dtype=np.float32)
        whisper_backend = backend.whisper if config.engine == "hybrid" else backend
        chunks, _ = whisper_backend._get_model().transcribe(
            probe, language="en", vad_filter=False, beam_size=1,
            word_timestamps=False, condition_on_previous_text=False,
        )
        list(chunks)

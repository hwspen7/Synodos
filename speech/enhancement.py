"""Audio quality estimates and optional offline denoising."""
from __future__ import annotations
import numpy as np


def quality(samples: np.ndarray) -> dict:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return {"rms": 0.0, "peak": 0.0, "clipped_ratio": 0.0, "silent": True}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return {"rms": rms, "peak": peak,
            "clipped_ratio": float(np.mean(np.abs(audio) > 0.99)),
            "silent": rms < 0.0005}


def optional_denoise(samples: np.ndarray, enabled: bool = False) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if not enabled or len(audio) < 4000:
        return audio
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise RuntimeError("MEETING_DENOISE requires noisereduce; install it before enabling") from exc
    return np.asarray(nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.6), dtype=np.float32)

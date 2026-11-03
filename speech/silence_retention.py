"""Conservative silence reclamation for an unconfirmed rolling audio buffer.

Never discard a sample with measurable voice-like energy. This is a fail-closed
optimization rather than a substitute for VAD or a disk-backed ingest queue.
"""
from __future__ import annotations

import numpy as np


def safe_silence_prefix(
    samples: np.ndarray,
    sample_rate: int = 16000,
    *,
    retain_seconds: float = 3.0,
    minimum_window_seconds: float = 7.0,
    frame_seconds: float = 0.20,
    max_peak: float = 0.025,
    max_rms: float = 0.003,
) -> int:
    """Number of oldest samples safe to reclaim after ASR returned no text.

    Every discarded frame must satisfy BOTH very-low peak and RMS thresholds.
    An incomplete frame is always retained. This deliberately prioritizes
    speech recall over preventing every possible overflow.
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1 or sample_rate <= 0 or not np.all(np.isfinite(arr)):
        return 0
    if len(arr) < int(minimum_window_seconds * sample_rate):
        return 0
    frame = max(1, int(frame_seconds * sample_rate))
    maximum = max(0, len(arr) - int(retain_seconds * sample_rate))
    end = 0
    while end + frame <= maximum:
        block = arr[end : end + frame]
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block * block)))
        if peak > max_peak or rms > max_rms:
            break
        end += frame
    return end

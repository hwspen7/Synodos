"""Conservative PCM noise veto for live caption confirmation.

Only obvious near-digital silence and steady, low-level noise are vetoed.
This is NOT a general speech detector: music, playback, transient noises
and voice-like ambience may pass. Never classify text by its words.
"""
from __future__ import annotations

import numpy as np


def uninformative_audio(audio: np.ndarray, sample_rate: int = 16000) -> bool:
    """Reject unmistakable quiet or stationary input, preserving uncertain voice.

    This guard operates exclusively in the live caption pipeline. Frozen ASR
    model benchmarks remain unchanged and cannot certify false-alarm rate.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        return False
    if not samples.size:
        return True
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    # Digital microphone silence and an extremely quiet electrical floor.
    if peak <= 0.003 and rms <= 0.0007:
        return True
    # Do not veto clear speech, sudden sound events, or short replies merely
    # because a model happens to distrust the content.
    if (sample_rate <= 0 or samples.size < round(sample_rate * 1.2)
            or peak > 0.05 or rms > 0.012):
        return False
    frame = max(1, round(sample_rate * 0.1))
    count = samples.size // frame
    if count < 12:
        return False
    frames = samples[: count * frame].reshape(count, frame).astype(np.float64)
    energies = np.sqrt(np.mean(frames * frames, axis=1))
    mean = float(np.mean(energies))
    # A continuous low-level fan/electronic noise floor varies little over
    # successive 100ms intervals; human syllables contain energy transitions.
    return bool(mean > 0 and
                float(np.std(energies) / mean) <= 0.08)

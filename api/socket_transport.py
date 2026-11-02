"""Validate streaming PCM payloads before sending them to audio buffers."""

from __future__ import annotations

import numpy as np

MAX_PCM_FRAME_BYTES = 1024 * 1024


def decode_pcm_frame(frame: bytes) -> np.ndarray:
    if not frame:
        raise ValueError("Empty PCM frame")
    if len(frame) % 2:
        raise ValueError("PCM frame length must be even")
    if len(frame) > MAX_PCM_FRAME_BYTES:
        raise ValueError("PCM frame exceeds the maximum size")
    return np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0

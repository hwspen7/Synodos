"""Optional independent WebRTC/Silero agreement gate for 16 kHz meeting audio.

Only overlap of two acoustic detectors is treated as verified speech.
This can reject weak noise false positives, but not voice-like playback,
and it can reject quiet/atypical real speech. Never infer text from VAD.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


SAMPLE_RATE = 16000
FRAME_MS = 30
PAD_MS = 120
MIN_WEBRTC_SUPPORT_MS = 210
MIN_WEBRTC_FRACTION = 0.20


def _merge(intervals: list[tuple[int, int]], *, gap: int = 0) -> list[tuple[int, int]]:
    result: list[list[int]] = []
    for a, b in sorted(intervals):
        if a >= b:
            continue
        if result and a <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return [(a, b) for a, b in result]


def crosscheck_speech_regions(
    samples: np.ndarray,
    silero_regions: Iterable[Mapping[str, int]],
    *,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = 2,
) -> list[dict[str, int]]:
    """Return audio regions supported by *both* Silero and WebRTC VAD.

    The VAD API receives contiguous 30 ms PCM16 frames. Regions returned
    use **sample offsets** (not milliseconds), matching faster-whisper.
    WebRTC speech flags are expanded by 120 ms to preserve phoneme edges;
    only Silero regions with >=210 ms of raw WebRTC speech are considered.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Dual VAD requires 16 kHz mono input")
    if not isinstance(aggressiveness, int) or aggressiveness not in (0, 1, 2, 3):
        raise ValueError("WebRTC VAD aggressiveness must be 0, 1, 2 or 3")
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "MEETING_WHISPER_DUAL_VAD=1 requires the webrtcvad-wheels package. "
            "Install requirements/speech-dual-vad.txt or disable dual VAD."
        ) from exc
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
        raise ValueError("Dual VAD requires a finite mono waveform")
    length = len(waveform)
    frame_samples = sample_rate * FRAME_MS // 1000
    if length < frame_samples:
        return []
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    detector = webrtcvad.Vad(aggressiveness)
    raw_web: list[tuple[int, int]] = []
    for a in range(0, length - frame_samples + 1, frame_samples):
        frame = pcm[a:a + frame_samples]
        if detector.is_speech(frame.tobytes(), sample_rate):
            raw_web.append((a, a + frame_samples))
    raw_web = _merge(raw_web)
    if not raw_web:
        return []
    pad = sample_rate * PAD_MS // 1000
    padded = _merge([(max(0, a - pad), min(length, b + pad)) for a, b in raw_web], gap=sample_rate * 150 // 1000)
    accepted: list[tuple[int, int]] = []
    for region in silero_regions:
        if not isinstance(region, Mapping):
            continue
        try:
            start = int(region["start"])
            end = int(region["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        start = min(length, max(start, 0))
        end = min(length, max(end, 0))
        if end <= start:
            continue
        web_support = sum(max(0, min(end, b) - max(start, a)) for a, b in raw_web)
        if (web_support * 1000 < MIN_WEBRTC_SUPPORT_MS * sample_rate or
                web_support / (end - start) < MIN_WEBRTC_FRACTION):
            continue
        for a, b in padded:
            x, y = max(start, a), min(end, b)
            if y > x:
                accepted.append((x, y))
    return [{"start": a, "end": b} for a, b in _merge(accepted)]

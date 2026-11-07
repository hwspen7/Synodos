"""Conservative, length-preserving audio conditioning for live ASR."""

from __future__ import annotations

import numpy as np


class AdaptiveAudio:
    """Process complete 20 ms blocks once per contiguous PCM stream.

    The caller supplies a rolling snapshot in absolute samples. Only the
    overlapping portion of the conditioned audio is cached. An incomplete
    trailing block is passed through so neither live nor final ASR loses it.
    """

    def __init__(self, sample_rate: int):
        self._block_size = max(1, round(sample_rate * 0.02))
        ramp = np.sin(np.linspace(0, np.pi / 2, max(1, round(sample_rate * 0.003)))) ** 2
        self._window = np.ones(self._block_size)
        self._window[:len(ramp)] = ramp
        self._window[-len(ramp):] = ramp[::-1]
        self._frequencies = np.fft.rfftfreq(self._block_size, 1 / sample_rate)
        self._gain_ramp = np.linspace(0, 1, self._block_size, dtype=np.float32)
        self._band_edges = np.arange(0, len(self._frequencies), 8)
        self._reset(0)

    def _reset(self, start: int) -> None:
        self._start = start
        self._cached = np.empty(0, dtype=np.float32)
        self._gain = 1.0
        self._quiet_blocks = 0
        self._clean_floor = False
        self._soft_blocks = 0
        self._strong_hold = 0
        self._candidate: np.ndarray | None = None
        self._candidate_blocks = 0
        self._noise: np.ndarray | None = None
        self._noise_age = 0

    def prepare(self, start: int, audio: np.ndarray) -> np.ndarray:
        end = start + len(audio)
        processed = self._start + len(self._cached)
        if start < self._start or start > processed or end < processed:
            self._reset(start)
            processed = start

        old = self._cached[start - self._start:]
        complete_end = processed + (end - processed) // self._block_size * self._block_size
        if complete_end > processed:
            conditioned = np.empty(complete_end - start, dtype=np.float32)
            conditioned[:len(old)] = old
            for offset in range(processed - start, len(conditioned), self._block_size):
                block = slice(offset, offset + self._block_size)
                conditioned[block] = self._condition(audio[block])
            self._cached = conditioned
        elif start != self._start:
            # Release samples reclaimed by the rolling input buffer.
            self._cached = old.copy()
        self._start = start

        if not len(self._cached):
            return audio
        if complete_end == end:
            return self._cached
        return np.concatenate((self._cached, audio[complete_end - start:]))

    def _condition(self, block: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block * block)))
        if not np.isfinite(peak) or peak >= 0.99:
            self._gain = 1.0
            self._quiet_blocks = 0
            self._soft_blocks = 0
            self._strong_hold = 30
            self._candidate = None
            self._candidate_blocks = 0
            return block

        quiet = rms < 0.0015 and peak < 0.006
        self._quiet_blocks = self._quiet_blocks + 1 if quiet else 0
        if self._quiet_blocks >= 12:
            self._clean_floor = True
            self._noise = None
        self._noise_age += 1
        if self._noise_age > 500:
            self._noise = None

        enhanced = block
        if not quiet and (0.005 <= rms < 0.08 or self._noise is not None):
            spectrum = np.fft.rfft(block * self._window)
            power = np.abs(spectrum) ** 2
            self._track_noise(rms, power)
            if self._noise is not None:
                # Subtract only the measured floor with a bounded mask.
                # The correction fades to zero at block boundaries.
                power_ratio = self._noise / np.maximum(power, 1e-12)
                mask = np.sqrt(np.maximum(0.36, 1.0 - 0.85 * power_ratio))
                correction = np.fft.irfft(spectrum * (mask - 1.0), n=len(block))
                enhanced = block + self._window * correction
        else:
            self._candidate = None
            self._candidate_blocks = 0

        # Normal-level syllables inhibit gain for the rest of the phrase.
        # Require sustained low level to avoid amplifying ordinary word tails.
        strong = rms >= 0.025 or peak >= 0.12
        self._strong_hold = 30 if strong else max(0, self._strong_hold - 1)
        soft = 0.003 < rms < 0.020 and not self._strong_hold
        self._soft_blocks = (min(12, self._soft_blocks + 1) if soft
                             else max(0, self._soft_blocks - 2))
        target = 1.0
        if soft and self._soft_blocks >= 8 and self._clean_floor and self._noise is None:
            target = min(2.0, 0.020 / rms)
        enhanced_peak = float(np.max(np.abs(enhanced)))
        if enhanced_peak >= 0.99:
            enhanced, enhanced_peak = block, peak
        max_gain = max(1.0, 0.95 / max(peak, enhanced_peak, 1e-6))
        initial = min(self._gain, max_gain)
        speed = 0.10 if target > initial else 0.40
        self._gain = min(max_gain, initial + speed * (target - initial))
        if abs(self._gain - 1.0) < 0.001:
            self._gain = 1.0
        if initial == self._gain == 1.0:
            return enhanced
        return enhanced * (initial + (self._gain - initial) * self._gain_ramp)

    def _track_noise(self, rms: float, power: np.ndarray) -> None:
        total = float(np.sum(power))
        dominant = int(np.argmax(power))
        # Sustained voiced notes are unsafe to learn as background. Allow
        # concentrated sub-100 Hz noise such as mains hum.
        if not 0.005 <= rms < 0.08 or (self._frequencies[dominant] >= 100 and power[dominant] > total * 0.35):
            self._candidate = None
            self._candidate_blocks = 0
            return
        previous = self._candidate
        # Check stationarity over broad bands. Individual FFT bins fluctuate
        # even for steady broadband noise.
        if previous is not None:
            current_bands = np.add.reduceat(power, self._band_edges)
            previous_bands = np.add.reduceat(previous, self._band_edges)
            stable = np.sum(np.abs(current_bands - previous_bands)) <= 0.60 * np.sum(previous_bands)
            if stable:
                self._candidate = previous * 0.88 + power * 0.12
                self._candidate_blocks += 1
                if self._candidate_blocks >= 16:
                    self._noise = self._candidate
                    self._noise_age = 0
                    self._clean_floor = False
                return
        self._candidate = power
        self._candidate_blocks = 1

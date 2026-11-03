"""Thread-safe rolling audio buffer.

- AudioBuffer holds the current window of unconfirmed audio.
    Confirmed text and embeddings are retained separately, not as raw audio.
"""

from __future__ import annotations

import threading
from typing import Tuple
from foundation.settings import load_infrastructure_settings
from observability.metrics import metrics

import numpy as np


class AudioBackpressureError(RuntimeError):
    """The unconfirmed audio capacity has been reached."""


class AudioBuffer:
    def __init__(self, sample_rate: int, max_seconds: int | None = None):
        self.sample_rate = sample_rate
        if max_seconds is None:
            max_seconds = load_infrastructure_settings().max_audio_buffer_seconds
        self._max_samples = int(sample_rate * max_seconds)
        self._chunks: list[np.ndarray] = []
        self._count = 0
        #: Absolute sample index of the first buffered sample.
        self._start_offset = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def append(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        with self._cond:
            if self._count + arr.size > self._max_samples:
                metrics.increment("audio_overflow_rejected")
                raise AudioBackpressureError("Audio processing is behind the recording stream")
            self._chunks.append(arr)
            self._count += arr.size
            metrics.gauge("audio_last_buffered_seconds", self._count / self.sample_rate)
            self._cond.notify_all()

    def seconds(self) -> float:
        with self._lock:
            return self._count / self.sample_rate

    def total_samples(self) -> int:
        # Keep the existing API contract: buffered sample count.
        with self._lock:
            return self._count

    def absolute_end_samples(self) -> int:
        # Monotonic sample endpoint for real-time preview timestamps.
        with self._lock:
            return self._start_offset + self._count

    def start_offset(self) -> int:
        with self._lock:
            return self._start_offset

    def set_start_offset(self, offset: int) -> None:
        """Set the global offset of the first sample when resuming a meeting."""
        with self._lock:
            self._start_offset = int(offset)

    def wait_until(
        self,
        min_seconds: float,
        stop_event: threading.Event,
        wake_event: threading.Event | None = None,
    ) -> bool:
        """Wait for the buffer to reach min_seconds or wake_event, returning False on stop."""
        with self._cond:
            while not stop_event.is_set():
                if wake_event is not None and wake_event.is_set():
                    return True
                if self._count / self.sample_rate >= min_seconds:
                    return True
                self._cond.wait(timeout=0.2)
        return False

    def wake(self) -> None:
        """Wake the processor thread, for example when stopping a session."""
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> Tuple[int, np.ndarray]:
        """Return the absolute starting sample index and a copy of buffered samples."""
        with self._lock:
            if not self._chunks:
                return self._start_offset, np.empty(0, dtype=np.float32)
            return self._start_offset, np.concatenate(self._chunks)

    def drop_before(self, drop_samples: int) -> None:
        """Drop the oldest drop_samples without affecting newly appended audio."""
        if drop_samples <= 0:
            return
        with self._cond:
            remaining = int(drop_samples)
            while remaining > 0 and self._chunks:
                first = self._chunks[0]
                if len(first) <= remaining:
                    self._chunks.pop(0)
                    self._count -= len(first)
                    self._start_offset += len(first)
                    remaining -= len(first)
                else:
                    self._chunks[0] = first[remaining:]
                    self._count -= remaining
                    self._start_offset += remaining
                    remaining = 0
            self._cond.notify_all()

    def clear(self) -> None:
        with self._cond:
            self._chunks.clear()
            self._count = 0
            self._cond.notify_all()

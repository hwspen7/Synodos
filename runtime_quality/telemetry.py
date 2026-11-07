"""Bounded, text-free ASR and GPU scheduling telemetry."""
from __future__ import annotations

import math
import threading
from collections import defaultdict, deque


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


class Phase07Telemetry:
    def __init__(self, capacity: int = 256):
        if not 1 <= capacity <= 4096:
            raise ValueError("Invalid telemetry capacity")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._wait: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=capacity))
        self._run: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=capacity))
        self._asr_rtf: deque[float] = deque(maxlen=capacity)
        self._queue_depth = 0
        self._last_buffered_s = 0.0

    def observe_queue_depth(self, count: int) -> None:
        with self._lock:
            self._queue_depth = max(0, int(count))

    def observe_queue_wait(self, operation: str, duration_s: float) -> None:
        if math.isfinite(duration_s) and duration_s >= 0:
            with self._lock:
                self._wait[str(operation)[:32]].append(duration_s * 1000)

    def observe_inference(self, operation: str, duration_s: float) -> None:
        if math.isfinite(duration_s) and duration_s >= 0:
            with self._lock:
                self._run[str(operation)[:32]].append(duration_s * 1000)

    def observe_asr_cycle(self, audio_s: float, duration_s: float) -> None:
        if 0 < audio_s <= 3600 and math.isfinite(duration_s) and duration_s >= 0:
            with self._lock:
                self._asr_rtf.append(duration_s / audio_s)
                self._last_buffered_s = float(audio_s)

    def snapshot(self) -> dict:
        with self._lock:
            operations = sorted(set(self._wait) | set(self._run))
            return {
                "queue_depth": self._queue_depth,
                "operations": {
                    op: {"queue_wait_p50_ms": percentile(list(self._wait[op]), 0.5),
                         "queue_wait_p95_ms": percentile(list(self._wait[op]), 0.95),
                         "run_p50_ms": percentile(list(self._run[op]), 0.5),
                         "run_p95_ms": percentile(list(self._run[op]), 0.95),
                         "observations": len(self._run[op])}
                    for op in operations
                },
                "asr_rtf_p50": percentile(list(self._asr_rtf), 0.5),
                "asr_rtf_p95": percentile(list(self._asr_rtf), 0.95),
                "last_asr_window_seconds": self._last_buffered_s,
            }


telemetry = Phase07Telemetry()

"""Bounded, priority-aware synchronous inference dispatcher.

A single daemon worker serializes GPU calls. Timeouts only abandon a caller's
wait and cannot safely preempt a running CUDA operation.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable, Any

from core.metrics import metrics
from runtime_quality.telemetry import telemetry


class InferenceBusyError(RuntimeError):
    pass


class InferenceTimeoutError(RuntimeError):
    pass


@dataclass(order=True)
class _Task:
    priority: int
    sequence: int
    submitted: float = field(compare=False)
    operation: str = field(compare=False)
    callback: Callable[[], Any] = field(compare=False)
    future: Future = field(compare=False)


class InferenceScheduler:
    def __init__(self, max_queue: int = 32, timeout_s: float = 180.0) -> None:
        if max_queue < 1 or timeout_s <= 0:
            raise ValueError("Invalid inference scheduler limits")
        self._pending: queue.PriorityQueue[_Task] = queue.PriorityQueue(maxsize=max_queue)
        self._sequence = itertools.count()
        self._closed = threading.Event()
        self.timeout_s = timeout_s
        self._worker = threading.Thread(target=self._run, daemon=True, name="model-inference")
        self._worker.start()

    def submit(self, operation: str, callback: Callable[[], Any], priority: int = 10,
               timeout_s: float | None = None) -> Any:
        if self._closed.is_set():
            raise InferenceBusyError("Inference scheduler is closed")
        future: Future = Future()
        task = _Task(priority, next(self._sequence), time.monotonic(), operation, callback, future)
        try:
            self._pending.put_nowait(task)
        except queue.Full as exc:
            metrics.increment("inference_rejected")
            raise InferenceBusyError("Inference queue is full") from exc
        metrics.gauge("inference_pending", self._pending.qsize())
        telemetry.observe_queue_depth(self._pending.qsize())
        try:
            return future.result(timeout=self.timeout_s if timeout_s is None else timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            metrics.increment("inference_wait_timeouts")
            raise InferenceTimeoutError("Inference request timed out") from exc

    def _run(self) -> None:
        while not self._closed.is_set() or not self._pending.empty():
            try:
                task = self._pending.get(timeout=0.1)
            except queue.Empty:
                continue
            metrics.gauge("inference_pending", self._pending.qsize())
            telemetry.observe_queue_depth(self._pending.qsize())
            try:
                if task.future.set_running_or_notify_cancel():
                    telemetry.observe_queue_wait(task.operation, time.monotonic() - task.submitted)
                    start = time.monotonic()
                    try:
                        result = task.callback()
                    except BaseException as exc:
                        task.future.set_exception(exc)
                        metrics.increment("inference_failed")
                    else:
                        task.future.set_result(result)
                        metrics.increment("inference_completed")
                    finally:
                        elapsed = time.monotonic() - start
                        metrics.gauge("inference_last_duration_s", elapsed)
                        telemetry.observe_inference(task.operation, elapsed)
            finally:
                self._pending.task_done()

    def close(self) -> None:
        self._closed.set()
        # Callers can still be inside a non-interruptible model operation.

    @property
    def pending(self) -> int:
        return self._pending.qsize()

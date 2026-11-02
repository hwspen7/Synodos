"""线程安全的音频缓冲。

- AudioBuffer：VAD 滑动窗口，只保留最近一段时间，处理器周期性对其做 VAD；
  已确认的文本与声纹不再重算，因此不需要保留全量音频。
"""

from __future__ import annotations

import threading
from typing import Tuple

import numpy as np


class AudioBuffer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._count = 0
        #: 缓冲区第一个样本在整个会话音频流中的绝对采样点序号
        self._start_offset = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def append(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        with self._cond:
            self._chunks.append(arr)
            self._count += arr.size
            self._cond.notify_all()

    def seconds(self) -> float:
        with self._lock:
            return self._count / self.sample_rate

    def total_samples(self) -> int:
        with self._lock:
            return self._count

    def start_offset(self) -> int:
        with self._lock:
            return self._start_offset

    def set_start_offset(self, offset: int) -> None:
        """设置窗口第一个样本的全局偏移（恢复已保存会议时使用）。"""
        with self._lock:
            self._start_offset = int(offset)

    def wait_until(
        self,
        min_seconds: float,
        stop_event: threading.Event,
        wake_event: threading.Event | None = None,
    ) -> bool:
        """等待缓冲区达到 min_seconds 秒（或 wake_event 置位）；被停止时返回 False。"""
        with self._cond:
            while not stop_event.is_set():
                if wake_event is not None and wake_event.is_set():
                    return True
                if self._count / self.sample_rate >= min_seconds:
                    return True
                self._cond.wait(timeout=0.2)
        return False

    def wake(self) -> None:
        """唤醒等待中的处理器线程（用于停止会话时）。"""
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> Tuple[int, np.ndarray]:
        """返回 (缓冲区起始绝对采样点, 全部样本的拷贝)。"""
        with self._lock:
            if not self._chunks:
                return self._start_offset, np.empty(0, dtype=np.float32)
            return self._start_offset, np.concatenate(self._chunks)

    def drop_before(self, drop_samples: int) -> None:
        """丢弃缓冲区最前面的 drop_samples 个样本（只从队首移除，不影响新追加的数据）。"""
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

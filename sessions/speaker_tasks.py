"""Globally bounded speaker analysis workers with ASR-first GPU scheduling."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import numpy as np

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="meeting-acoustic-speakers")
_SLOTS = threading.BoundedSemaphore(24)


def submit_speaker_analysis(sentence: dict, samples: np.ndarray, embed, sample_rate: int) -> Future | None:
    """Queue isolated audio analysis. No text is discarded when at capacity."""
    if not _SLOTS.acquire(blocking=False):
        return None
    audio = np.asarray(samples, dtype=np.float32).copy()
    fields = dict(sentence)

    def work():
        try:
            from speech.turn_split import split_by_acoustic_turns
            return split_by_acoustic_turns(fields, audio, embed, sample_rate=sample_rate)
        finally:
            _SLOTS.release()
    try:
        return _POOL.submit(work)
    except BaseException:
        _SLOTS.release()
        raise

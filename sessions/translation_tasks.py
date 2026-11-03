"""Bounded translation workers detached from ASR and speaker inference."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="meeting-translation")
_SLOTS = threading.BoundedSemaphore(32)


def submit_translation(text: str, source: str, target: str) -> Future | None:
    """Schedule a language pair; None means the bounded queue is saturated."""
    if not _SLOTS.acquire(blocking=False):
        return None

    def task() -> str:
        from speech.translation import translate
        try:
            return translate(text, source, target)
        finally:
            _SLOTS.release()

    try:
        return _POOL.submit(task)
    except BaseException:
        _SLOTS.release()
        raise

"""Bounded caption translation, independent of speech and speaker inference."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading

from speech.captions.translation import translate_batch

BATCH_SIZE = 6
MAX_PENDING = 48
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meeting-translation")
_SLOTS = threading.BoundedSemaphore(8)


@dataclass(frozen=True)
class CaptionTranslation:
    id: int
    text: str
    source: str

    @classmethod
    def from_segment(cls, segment: dict) -> CaptionTranslation:
        return cls(int(segment["id"]), str(segment.get("text") or ""),
                   str(segment.get("language") or "und"))

    def apply(self, segment: dict, target: str, text: str) -> bool:
        if (self != self.from_segment(segment) or not text.strip()
                or (segment.get("translation_target") == target and segment.get("translation"))):
            return False
        segment.update(translation=text.strip(), translation_target=target)
        return True


def caption_batch(captions: Iterable[CaptionTranslation]) -> tuple[CaptionTranslation, ...]:
    """Limit prompt size without splitting a confirmed caption."""
    batch = []
    remaining = 2400
    for caption in captions:
        if batch and (len(batch) == BATCH_SIZE or len(caption.text) > remaining):
            break
        batch.append(caption)
        remaining -= len(caption.text)
    return tuple(batch)


def submit_translation(captions: tuple[CaptionTranslation, ...], target: str) -> Future | None:
    """None means the shared queue is full. Cancelled work frees its slot."""
    if not 1 <= len(captions) <= BATCH_SIZE:
        raise ValueError("Invalid caption translation batch size")
    if not _SLOTS.acquire(blocking=False):
        return None
    try:
        future = _POOL.submit(translate_batch, tuple(
            (caption.id, caption.text, caption.source) for caption in captions
        ), target)
    except BaseException:
        _SLOTS.release()
        raise
    future.add_done_callback(lambda _: _SLOTS.release())
    return future

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
            from sessions.speaker_fallback import normalized_embedding
            pieces = split_by_acoustic_turns(fields, audio, embed, sample_rate=sample_rate)
            if (len(pieces) == 1 and pieces[0].get("emb") is None
                    and not pieces[0].get("ambiguous")
                    and len(audio) >= int(sample_rate * 1.2)):
                try:
                    pieces[0]["emb"] = normalized_embedding(embed(audio))
                except Exception:
                    pieces[0]["ambiguous"] = True
            return pieces
        finally:
            _SLOTS.release()
    try:
        return _POOL.submit(work)
    except BaseException:
        _SLOTS.release()
        raise

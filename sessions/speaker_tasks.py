"""Globally bounded speaker analysis workers with ASR-first GPU scheduling."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import numpy as np

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="meeting-acoustic-speakers")
_SLOTS = threading.BoundedSemaphore(24)


def submit_speaker_analysis(
    sentence: dict, samples: np.ndarray, embed, sample_rate: int,
    *, minimum_embedding_seconds: float = 1.2,
) -> Future | None:
    if not 0.8 <= minimum_embedding_seconds <= 1.2:
        raise ValueError("Speaker embedding duration floor must be 0.8..1.2s")
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
                    and len(audio) >= int(sample_rate * minimum_embedding_seconds)):
                try:
                    pieces[0]["emb"] = normalized_embedding(embed(audio))
                    if len(audio) < int(sample_rate * 1.2):
                        # Explicit candidate under the normal safe floor.
                        # Such samples must not seed trusted ECAPA prototypes.
                        pieces[0]["short_probe"] = True
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



def submit_acoustic_window(
    start_sample: int, samples: np.ndarray, embed, sample_rate: int,
) -> Future | None:
    """Analyze independent, bounded voice windows without holding up ASR."""
    if not _SLOTS.acquire(blocking=False):
        return None
    audio = np.asarray(samples, dtype=np.float32).copy()

    def work():
        try:
            # Cheap silence gate avoids embedding non-voice audio. It is not
            # a speech classifier and must not discard the audio itself.
            if len(audio) < int(sample_rate * 1.2):
                return start_sample, len(audio), None
            rms = float(np.sqrt(np.mean(audio * audio)))
            if not np.isfinite(rms) or rms < 0.001:
                return start_sample, len(audio), None
            from speech.acoustic_timeline import normalize
            return start_sample, len(audio), normalize(embed(audio))
        finally:
            _SLOTS.release()

    try:
        return _POOL.submit(work)
    except BaseException:
        _SLOTS.release()
        raise

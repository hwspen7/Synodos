"""Conservative acoustic speaker-turn partitioning of ASR word intervals.

Voice changes are measured from acoustic embeddings on independent windows,
not inferred from the language or wording of the transcript.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import math
import os
import unicodedata
from collections.abc import Callable

import numpy as np
from speech.text_format import join_transcript_text


def _vector(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        a = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    n = float(np.linalg.norm(a))
    return a / n if a.size and np.all(np.isfinite(a)) and n > 1e-7 else None


def _letters(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def split_by_acoustic_turns(
    sentence: dict,
    samples: np.ndarray,
    embed: Callable[[np.ndarray], object],
    *,
    sample_rate: int = 16000,
    min_window_s: float = 1.4,
    max_windows: int = 8,
    similarity: float | None = None,
) -> list[dict]:
    """Return timed word groups. Mixed speech without usable word times is unassigned.

    The function never invents a word or an identity. Failure to obtain reliable
    embeddings leaves the original segment intact for the calling pipeline.
    """
    start, end = int(sentence["start"]), int(sentence["end"])
    original = str(sentence.get("text", "") or "")
    fallback = [{"start": start, "end": end, "text": original,
                 "emb": None, "ambiguous": False}]
    if sample_rate < 1000 or not original.strip() or end <= start:
        return fallback
    n_samples = len(samples)
    count = min(max_windows, int(n_samples // (sample_rate * min_window_s)))
    if count < 2:
        return fallback
    threshold = float(os.getenv("MEETING_SPEAKER_TURN_SIMILARITY", "0.58")) if similarity is None else similarity
    if not (math.isfinite(threshold) and 0.1 <= threshold <= 0.95):
        raise ValueError("Invalid speaker-turn similarity threshold")
    windows: list[tuple[int, int, np.ndarray]] = []
    for i in range(count):
        a = i * n_samples // count
        b = (i + 1) * n_samples // count
        try:
            vec = _vector(embed(samples[a:b]))
        except Exception:
            return fallback
        if vec is None:
            return fallback
        windows.append((a, b, vec))
    changes = [i for i in range(1, len(windows))
               if float(np.dot(windows[i-1][2], windows[i][2])) < threshold]
    if not changes:
        all_vectors = np.stack([v for _, _, v in windows])
        return [dict(fallback[0], emb=_vector(all_vectors.mean(axis=0)))]
    words = sentence.get("words", [])
    if not isinstance(words, list) or len(words) < 2:
        return [dict(fallback[0], ambiguous=True)]
    valid = []
    for word in words:
        if not isinstance(word, dict):
            return [dict(fallback[0], ambiguous=True)]
        try:
            a, b = int(word["start_ms"]), int(word["end_ms"])
            value = str(word.get("text", "") or "")
        except (KeyError, ValueError, TypeError):
            return [dict(fallback[0], ambiguous=True)]
        if a < start or b > end or b <= a or not value.strip():
            return [dict(fallback[0], ambiguous=True)]
        valid.append((a, b, value))
    joined = ""
    for _, _, t in valid:
        joined = join_transcript_text(joined, t)
    source, reconstructed = _letters(original), _letters(joined)
    if not source or SequenceMatcher(None, source, reconstructed, autojunk=False).ratio() < 0.91:
        return [dict(fallback[0], ambiguous=True)]
    cuts = [start + int(windows[index][0] * 1000 / sample_rate) for index in changes]
    groups: list[list[tuple[int, int, str]]] = []
    for a, b, t in valid:
        mid = (a + b) / 2
        label = sum(mid >= cut for cut in cuts)
        while len(groups) <= label:
            groups.append([])
        groups[label].append((a, b, t))
    result = []
    for group in groups:
        if not group:
            continue
        a, b = group[0][0], group[-1][1]
        words_text = ""
        for _, _, text in group:
            words_text = join_transcript_text(words_text, text)
        related = [v for x, y, v in windows
                   if start + y * 1000 / sample_rate > a and start + x * 1000 / sample_rate < b]
        vec = _vector(np.stack(related).mean(axis=0)) if related and b-a >= 1200 else None
        result.append({"start": a, "end": b, "text": words_text,
                       "emb": vec, "ambiguous": vec is None})
    if len(result) < 2:
        return [dict(fallback[0], ambiguous=True)]
    return result

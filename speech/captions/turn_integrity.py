"""Validate asynchronous speaker-turn partitions before mutating confirmed text.

Speaker analysis is advisory: an invalid partition may never replace a
confirmed ASR hypothesis, invent text, or expand time outside the source turn.
"""
from __future__ import annotations

from collections.abc import Mapping
import unicodedata


def _surface(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", value)
        if not c.isspace()
    )


def valid_turn_partition(
    start_ms: int,
    end_ms: int,
    confirmed_text: str,
    pieces: object,
    *,
    timestamp_tolerance_ms: int = 100,
) -> bool:
    """Accept only ordered, text-conserving, source-bounded turns.

    A model output is not a trusted timestamp or text authority. Whitespace
    may change during alignment, but *all* source characters must survive.
    The tolerance handles quantized word-boundary timestamps, not arbitrary
    overlaps or speaker channels.
    """
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or end_ms <= start_ms
            or not isinstance(confirmed_text, str)
            or not confirmed_text.strip()
            or not isinstance(pieces, (list, tuple)) or not pieces
            or not 0 <= timestamp_tolerance_ms <= 250):
        return False
    texts: list[str] = []
    previous_end: int | None = None
    for piece in pieces:
        if not isinstance(piece, Mapping):
            return False
        start, end, text = piece.get("start"), piece.get("end"), piece.get("text")
        if (type(start) is not int or type(end) is not int
                or not isinstance(text, str) or not text.strip()
                or end <= start or start < start_ms - timestamp_tolerance_ms
                or end > end_ms + timestamp_tolerance_ms):
            return False
        if previous_end is not None and start < previous_end - timestamp_tolerance_ms:
            return False
        previous_end = end
        texts.append(text)
    return _surface("".join(texts)) == _surface(confirmed_text)

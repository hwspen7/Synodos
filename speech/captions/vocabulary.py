"""Keep meeting vocabulary intact and detect prompted decoding loops."""

from __future__ import annotations

import re
import unicodedata

from speech.contracts import Utterance

MAX_PROMPT_CHARS = 1500
_SEPARATORS = re.compile(r"[,;\n\r\uFF0C\u3001\uFF1B]+")


def normalize_terms(entries) -> list[str]:
    """Split explicit delimiters while retaining whitespace inside phrases."""
    if isinstance(entries, str):
        entries = [entries]
    seen = {}
    for entry in entries or ():
        if not isinstance(entry, str):
            continue
        for part in _SEPARATORS.split(entry):
            term = " ".join(part.split())
            if term:
                seen[term] = None
    return list(seen)


def prompt_terms(entries) -> list[str]:
    """Use complete phrases only, even when a library exceeds the budget."""
    result = []
    remaining = MAX_PROMPT_CHARS
    for term in normalize_terms(entries):
        cost = len(term) + (2 if result else 0)
        if cost <= remaining:
            result.append(term)
            remaining -= cost
    return result


def _compact(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", text).casefold()
        if char.isalnum()
    )


def repeated_terms(utterances: tuple[Utterance, ...], terms: list[str]) -> set[str]:
    """Find three close occurrences of the same prompted term in a speech run."""
    runs = []
    pieces = []
    end_ms = None
    for item in sorted(utterances, key=lambda item: item.start_ms):
        if end_ms is not None and item.start_ms > end_ms + 450:
            runs.append("".join(pieces))
            pieces = []
        pieces.append(_compact(item.text))
        end_ms = max(end_ms or 0, item.end_ms)
    runs.append("".join(pieces))

    repeated = set()
    for term in terms:
        key = _compact(term)
        if len(key) < 3:
            continue
        pattern = re.compile(r"(?:%s.{0,2}?){2}%s" % (re.escape(key), re.escape(key)))
        if any(pattern.search(run) for run in runs):
            repeated.add(key)
    return repeated


def unaffected_utterances(utterances: tuple[Utterance, ...], repeated: set[str]):
    """Return other utterances whose coverage must survive an unbiased retry."""
    unaffected = []
    for item in utterances:
        compact = _compact(item.text)
        if not any(key in compact for key in repeated):
            unaffected.append(item)
    return tuple(unaffected)

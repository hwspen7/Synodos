"""Conservative sentence units from authenticated ASR word timestamps.

The source ASR remains the authority. Only split when complete, ordered
per-word alignment reconstructs the recognized words. The splitter does not
invent text, speaker identities, or timings. No word alignment means no
synthetic timing: the frontend may still display soft line breaks.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from speech.contracts import Utterance, Word
from speech.text_format import join_transcript_text

_FINAL = frozenset(".!?。！？")
_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
})
_NUMBER = re.compile(r"^[0-9]+(?:[,.][0-9]+)*[.]?$")


def _characters(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _trailing_terminal(word: str) -> bool:
    text = word.strip().rstrip('"”’»)]}）』」')
    if not text or text.casefold() in _ABBREVIATIONS or _NUMBER.fullmatch(text):
        return False
    return text[-1] in _FINAL


def _joined(words: list[Word]) -> str:
    text = ""
    for word in words:
        text = join_transcript_text(text, word.text)
    return text.strip()


def split_utterance(
    utterance: Utterance, *,
    silence_ms: int = 780,
    min_unit_ms: int = 260,
    max_unit_ms: int = 8500,
    max_chars: int = 220,
) -> tuple[Utterance, ...]:
    """Partition long Whisper spans into stable, timestamped conversational units.

    Boundaries use punctuation, actual word gaps and a maximum readable length.
    Last words of abbreviations and decimal values are not treated as endings.
    A missing or inconsistent alignment conservatively returns the input.
    """
    words = list(utterance.words)
    if len(words) < 2 or utterance.end_ms <= utterance.start_ms:
        return (utterance,)
    if (any(not w.text.strip() or w.end_ms <= w.start_ms for w in words) or
            any(w.start_ms < utterance.start_ms - 150 or
                w.end_ms > utterance.end_ms + 150 for w in words) or
            any(words[i].start_ms < words[i-1].start_ms for i in range(1, len(words)))):
        return (utterance,)
    rebuilt = _joined(words)
    old = _characters(utterance.text)
    new = _characters(rebuilt)
    if not old or not new or SequenceMatcher(None, old, new, autojunk=False).ratio() < 0.94:
        return (utterance,)
    groups: list[list[Word]] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        current.append(word)
        if i == len(words) - 1:
            groups.append(current)
            break
        next_word = words[i + 1]
        duration = word.end_ms - current[0].start_ms
        gap = next_word.start_ms - word.end_ms
        ending = _trailing_terminal(word.text)
        too_long = duration >= max_unit_ms or len(_joined(current)) >= max_chars
        boundary = ((ending and duration >= min_unit_ms)
                    or (gap >= silence_ms and duration >= min_unit_ms)
                    or too_long)
        if boundary:
            groups.append(current)
            current = []
    if len(groups) <= 1:
        return (utterance,)
    output: list[Utterance] = []
    for group in groups:
        text = _joined(group)
        if not text:
            return (utterance,)
        output.append(Utterance(
            text, group[0].start_ms, group[-1].end_ms, utterance.language,
            tuple(group), "word", utterance.confidence, utterance.language_confidence,
        ))
    if _characters(" ".join(s.text for s in output)) != _characters(rebuilt):
        return (utterance,)
    return tuple(output)


def split_utterances(items: tuple[Utterance, ...]) -> tuple[Utterance, ...]:
    result: list[Utterance] = []
    for item in items:
        result.extend(split_utterance(item))
    return tuple(result)

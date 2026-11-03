"""Language-neutral joining of independently confirmed transcript utterances."""
from __future__ import annotations

import unicodedata

NO_SPACE_BEFORE = frozenset(",.!?;:%)]}，。！？；：、）》」』”’")
NO_SPACE_AFTER = frozenset("([{（《「『“‘")


def _cjk(char: str) -> bool:
    label = unicodedata.name(char, "")
    return "CJK" in label or "HIRAGANA" in label or "KATAKANA" in label or "HANGUL" in label


def join_transcript_text(left: str, right: str) -> str:
    """Preserve CJK adjacency but separate independent Latin-script utterances."""
    left, right = str(left or "").rstrip(), str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    a, b = left[-1], right[0]
    if a in NO_SPACE_AFTER or b in NO_SPACE_BEFORE or _cjk(a) or _cjk(b):
        return left + right
    return left + " " + right

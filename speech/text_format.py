"""Language-neutral joining of independently confirmed transcript utterances."""
from __future__ import annotations

import unicodedata

NO_SPACE_BEFORE = frozenset(",.!?;:%)]}\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u300b\u300d\u300f\u201d\u2019")
NO_SPACE_AFTER = frozenset("([{\uff08\u300a\u300c\u300e\u201c\u2018")


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

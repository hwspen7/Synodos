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



def split_display_sentences(text: str) -> list[str]:
    """Format punctuation-delimited utterances without touching source storage.

    This fallback is for legacy segments without reliable word timing; it does
    not manufacture per-sentence timestamps or identities.
    """
    import re

    value = str(text or "").strip()
    if not value:
        return []
    abbreviations = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
    }
    result = []
    current = ""
    for i, char in enumerate(value):
        current += char
        if char not in ".!?\u3002\uff01\uff1f":
            continue
        cjk = char in "\u3002\uff01\uff1f"
        following = value[i + 1] if i + 1 < len(value) else ""
        if not cjk and following and not following.isspace() and following not in '"\u201d\u2019':
            continue
        token = re.search(r"(?:^|\s)(\S+)$", current)
        if not cjk and token and token.group(1).casefold() in abbreviations:
            continue
        if not cjk and char == "." and current[:-1].endswith(tuple("0123456789")) and following.isdigit():
            continue
        if following in '"\u201d\u2019':
            # The closing quote is retained with its spoken sentence.
            continue
        if current.strip():
            result.append(current.strip())
        current = ""
    if current.strip():
        result.append(current.strip())
    return result or [value]


def ends_finished_thought(text: str) -> bool:
    """Detect completed sentence punctuation when deciding whether to merge."""
    import re
    return bool(re.search(r'[.!?\u3002\uff01\uff1f]["\u201d\u2019)\]]*$', str(text or "").strip()))

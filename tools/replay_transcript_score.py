"""Local-only, reference-backed text scoring for live caption replay.

This module never downloads a reference, guesses a speaker transcript, or
writes recognized words into a report. Inputs are supplied explicitly by
the local operator and discarded after aggregate metrics are computed.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import unicodedata

from speech.text_format import join_transcript_text


def finalized_transcript(state: Mapping) -> str:
    """Recover the final confirmed text in audio order, not segment ID order."""
    transcripts = state.get("segment_texts", {})
    order = state.get("segment_order", [])
    if not isinstance(transcripts, dict) or not isinstance(order, list):
        raise ValueError("Missing final transcript snapshot")
    if len(order) != len(transcripts):
        raise ValueError("Final segment order is inconsistent")
    combined = ""
    seen = set()
    for segment_id in order:
        if type(segment_id) is not int or segment_id in seen:
            raise ValueError("Invalid or repeated final segment ID")
        seen.add(segment_id)
        if segment_id not in transcripts:
            raise ValueError("Final segment text is missing")
        combined = join_transcript_text(
            combined, str(transcripts[segment_id] or ""))
    return combined


def _normal(text: str, *, include_punctuation: bool = False) -> str:
    """Case-fold and NFKC-normalize; optionally keep punctuation."""
    value = unicodedata.normalize("NFKC", text).casefold()
    characters = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("Z") or char.isspace():
            continue
        if not include_punctuation and category.startswith("P"):
            continue
        characters.append(char)
    return "".join(characters)


def _word_tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(" " if unicodedata.category(c).startswith("P")
                    else c for c in value)
    return [token for token in value.split() if token]


def edit_distance(left: list | str, right: list | str) -> int:
    """Exact Levenshtein distance using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, token in enumerate(left, 1):
        current = [index]
        for position, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[position] + 1,
                previous[position - 1] + (token != other),
            ))
        previous = current
    return previous[-1]


def score_reference(reference: str, hypothesis: str) -> dict:
    """Aggregate CER and English whitespace-token WER; never emit raw text.

    CER excludes punctuation and whitespace after NFKC/casefold; the
    separate punctuation-sensitive CER keeps punctuation. WER is only
    reported when the reference is compatible with space-delimited words.
    These are linear-transcript metrics, not speaker-turn/overlap DER.
    """
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("Reference and hypothesis must be UTF-8 text")
    ref_chars = _normal(reference)
    hyp_chars = _normal(hypothesis)
    if not ref_chars:
        raise ValueError("Reference contains no scoreable characters")
    ref_full = _normal(reference, include_punctuation=True)
    hyp_full = _normal(hypothesis, include_punctuation=True)
    char_errors = edit_distance(ref_chars, hyp_chars)
    punct_errors = edit_distance(ref_full, hyp_full)
    word_refs = _word_tokens(reference)
    word_hyps = _word_tokens(hypothesis)
    # Whitespace-token WER is not meaningful for unsegmented CJK text.
    cjk = re.search(r"[\u2e80-\u9fff\uf900-\ufaff]", reference) is not None
    has_word_spaces = bool(re.search(r"\s", reference.strip()))
    word_evaluable = not cjk and has_word_spaces and bool(word_refs)
    wer_errors = edit_distance(word_refs, word_hyps) if word_evaluable else None
    wer = round(wer_errors / len(word_refs), 4) if wer_errors is not None else None
    return {
        "reference_characters": len(ref_chars),
        "hypothesis_characters": len(hyp_chars),
        "character_edits": char_errors,
        "cer": round(char_errors / len(ref_chars), 4),
        "cer_percent": round(100 * char_errors / len(ref_chars), 2),
        "punctuation_sensitive_cer_percent": (
            round(100 * punct_errors / len(ref_full), 2)
            if ref_full else None
        ),
        "reference_words": len(word_refs) if word_evaluable else None,
        "hypothesis_words": len(word_hyps) if word_evaluable else None,
        "word_edits": wer_errors,
        "wer": wer,
        "wer_percent": round(100 * wer, 2) if wer is not None else None,
        "note": (
            "Local full-transcript reference scoring, NFKC/casefold; "
            "CER ignores punctuation and whitespace; WER uses space-delimited "
            "tokens when supported. Does not score diarization, speaker turns, "
            "overlap, timing, or independently certify reference correctness."
        ),
    }


def write_private_transcript(path: Path, text: str) -> None:
    """Opt-in export strictly below gitignored benchmark/fixed195."""
    safe = Path("benchmark/fixed195").resolve()
    target = path.resolve()
    if not target.is_relative_to(safe) or target.suffix.lower() != ".txt":
        raise ValueError(
            "Private transcript export requires a .txt path inside "
            "benchmark/fixed195 (gitignored)")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing local transcript audit file")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")

"""Local-only, reference-backed text scoring for live caption replay.

This module never downloads a reference, guesses a speaker transcript, or
writes recognized words into a report. Inputs are supplied explicitly by
the local operator and discarded after aggregate metrics are computed.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import argparse
import json
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


def alignment_error_counts(reference: list | str, hypothesis: list | str,
                           *, max_cells: int = 16_000_000) -> dict | None:
    """Counts from one deterministic minimum-edit alignment.

    The reference is the expected transcript. Deletion means content in
    reference but absent from hypothesis; insertion means extra hypothesis
    content. Equally optimal alignments can have different operation counts,
    so these are review indicators, never proof of word-level causation.
    Memory is O(min(reference_length, hypothesis_length)). A bounded
    optional audit never prevents the existing CER/WER scorer from working.
    """
    r, h = list(reference), list(hypothesis)
    if not r and not h:
        return {"deletions": 0, "insertions": 0, "substitutions": 0, "edits": 0}
    if len(r) * len(h) > max_cells:
        return None
    # Smaller dimension is across columns. Transposition reverses the
    # meaning of insertion/deletion, repaired in the returned counters.
    transposed = len(h) > len(r)
    outer, inner = (h, r) if transposed else (r, h)
    # (distance, substitutions, deletions, insertions)
    previous = [(i, 0, 0, i) for i in range(len(inner) + 1)]
    for i, expected in enumerate(outer, 1):
        current = [(i, 0, i, 0)]
        for j, observed in enumerate(inner, 1):
            diag = previous[j - 1]
            if expected == observed:
                best = diag
            else:
                best = (diag[0] + 1, diag[1] + 1, diag[2], diag[3])
            above = previous[j]
            deletion = (above[0] + 1, above[1], above[2] + 1, above[3])
            if deletion[0] < best[0]:
                best = deletion
            left = current[j - 1]
            insertion = (left[0] + 1, left[1], left[2], left[3] + 1)
            if insertion[0] < best[0]:
                best = insertion
            current.append(best)
        previous = current
    edits, substitutions, deletions, insertions = previous[-1]
    if transposed:
        deletions, insertions = insertions, deletions
    if edits != substitutions + deletions + insertions:
        raise AssertionError("Edit operation conservation failed")
    return {
        "deletions": deletions, "insertions": insertions,
        "substitutions": substitutions, "edits": edits,
    }


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
    # This is not a second CER/WER algorithm. Operation breakdown uses an
    # optimal alignment to explain the already-reported Levenshtein edits.
    char_ops = alignment_error_counts(ref_chars, hyp_chars)
    word_ops = (alignment_error_counts(word_refs, word_hyps)
                if word_evaluable else None)
    if ((char_ops is not None and char_ops["edits"] != char_errors)
            or (word_ops is not None and word_ops["edits"] != wer_errors)):
        raise AssertionError("Text alignment disagrees with Levenshtein score")
    return {
        "reference_characters": len(ref_chars),
        "hypothesis_characters": len(hyp_chars),
        "character_edits": char_errors,
        "character_deletions": (
            char_ops["deletions"] if char_ops is not None else None
        ),
        "character_insertions": (
            char_ops["insertions"] if char_ops is not None else None
        ),
        "character_substitutions": (
            char_ops["substitutions"] if char_ops is not None else None
        ),
        "reference_character_deletion_rate": (
            round(char_ops["deletions"] / len(ref_chars), 4)
            if char_ops is not None else None
        ),
        "cer": round(char_errors / len(ref_chars), 4),
        "cer_percent": round(100 * char_errors / len(ref_chars), 2),
        "punctuation_sensitive_cer_percent": (
            round(100 * punct_errors / len(ref_full), 2)
            if ref_full else None
        ),
        "reference_words": len(word_refs) if word_evaluable else None,
        "hypothesis_words": len(word_hyps) if word_evaluable else None,
        "word_edits": wer_errors,
        "word_deletions": (
            word_ops["deletions"] if word_ops is not None else None
        ),
        "word_insertions": (
            word_ops["insertions"] if word_ops is not None else None
        ),
        "word_substitutions": (
            word_ops["substitutions"] if word_ops is not None else None
        ),
        "reference_word_deletion_rate": (
            round(word_ops["deletions"] / len(word_refs), 4)
            if word_ops is not None else None
        ),
        "wer": wer,
        "wer_percent": round(100 * wer, 2) if wer is not None else None,
        "note": (
            "Local full-transcript reference scoring, NFKC/casefold; "
            "CER ignores punctuation and whitespace; WER uses space-delimited "
            "tokens when supported. Does not score diarization, speaker turns, "
            "overlap, timing, or independently certify reference correctness. "
            "Deletions, insertions and substitutions come from one optimal "
            "edit alignment and can differ across equal-score alignments. "
            "Null operation breakdown means the comparison exceeded a "
            "bounded computational budget; the base CER/WER remain intact."
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



def main() -> int:
    """Score the exact saved local replay text without replaying the WAV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-text", type=Path, required=True,
                        help="Opt-in private ASR transcript exported by replay")
    parser.add_argument("--reference-text", type=Path, required=True,
                        help="Independently checked human transcription of same WAV")
    parser.add_argument("--output", type=Path,
                        help="Optional local aggregate-only JSON report")
    args = parser.parse_args()
    if args.hypothesis_text.resolve() == args.reference_text.resolve():
        parser.error("Hypothesis and independent reference must be different files")
    reference = args.reference_text.read_text(encoding="utf-8-sig")
    hypothesis = args.hypothesis_text.read_text(encoding="utf-8-sig")
    metrics = score_reference(reference, hypothesis)
    report = {
        "schema": "meeting-live-caption-local-text-score-v1",
        **metrics,
        "reference_verified_by_user": "not_checked_by_tool",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                               + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

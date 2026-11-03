"""Privacy-preserving metrics for multilingual transcription benchmarks.

The functions take text in memory but return scores only. No model or GPU needed.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in value)
    return " ".join(value.split())


def char_tokens(value: str) -> list[str]:
    return [ch for ch in normalize_text(value) if not ch.isspace()]


def mixed_tokens(value: str) -> list[str]:
    """Use one token per CJK ideograph and whitespace-delimited Latin words."""
    normal = normalize_text(value)
    return re.findall(r"[\u3400-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]|[^\s\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", normal)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score_pair(reference: str, hypothesis: str, *, terms: list[str] | None = None,
               language: str | None = None) -> dict:
    ref_chars, hyp_chars = char_tokens(reference), char_tokens(hypothesis)
    # Whitespace tokenization is not valid word segmentation for Thai.
    # Reject an unsupported metric instead of implying a measured Thai WER.
    token_supported = language not in ("th",)
    ref_words = mixed_tokens(reference) if token_supported else []
    hyp_words = mixed_tokens(hypothesis) if token_supported else []
    c_err = edit_distance(ref_chars, hyp_chars)
    w_err = edit_distance(ref_words, hyp_words) if token_supported else 0
    ref_normal, hyp_normal = normalize_text(reference), normalize_text(hypothesis)
    matched_terms = [term for term in (terms or []) if normalize_text(term) in ref_normal]
    hit_terms = sum(normalize_text(term) in hyp_normal for term in matched_terms)
    return {
        "reference_characters": len(ref_chars), "character_edits": c_err,
        "cer": c_err / max(len(ref_chars), 1),
        "reference_tokens": len(ref_words), "token_edits": w_err,
        "token_metric_supported": token_supported,
        "mixed_token_error_rate": (w_err / len(ref_words) if ref_words else None),
        "glossary_terms": len(matched_terms), "glossary_hits": hit_terms,
    }


def aggregate(rows: list[dict]) -> dict:
    total = Counter()
    latencies = []
    language_evaluable = 0
    language_correct = 0
    failures = 0
    for item in rows:
        if item.get("error"):
            failures += 1
            continue
        for field in ("reference_characters", "character_edits", "reference_tokens", "token_edits",
                      "glossary_terms", "glossary_hits"):
            total[field] += int(item.get(field, 0))
        if isinstance(item.get("latency_ms"), (int, float)):
            latencies.append(float(item["latency_ms"]))
        if item.get("expected_language") and item.get("predicted_language"):
            language_evaluable += 1
            language_correct += item["expected_language"] == item["predicted_language"]
    result = {
        "samples": len(rows), "failed": failures, "evaluated": len(rows) - failures,
        # Do not report zero errors when every model invocation failed.
        "cer": (total["character_edits"] / total["reference_characters"]
                if total["reference_characters"] else None),
        "mixed_token_error_rate": (total["token_edits"] / total["reference_tokens"]
                                  if total["reference_tokens"] else None),
        "glossary_recall": (total["glossary_hits"] / total["glossary_terms"]
                            if total["glossary_terms"] else None),
        "language_accuracy": (language_correct / language_evaluable if language_evaluable else None),
        "mean_latency_ms": (round(sum(latencies) / len(latencies), 2) if latencies else None),
    }
    return result

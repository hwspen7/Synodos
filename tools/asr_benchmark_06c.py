"""Evaluate ASR hypotheses against manually reviewed references.

Input JSON is a list of objects with reference, hypothesis and optional
reference_language / detected_language. No audio or user data is uploaded.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, x in enumerate(left, 1):
        curr = [i]
        for j, y in enumerate(right, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (x != y)))
        prev = curr
    return prev[-1]


def score(records: list[dict]) -> dict:
    cer_errors = cer_refs = wer_errors = wer_refs = 0
    lid_correct = lid_total = 0
    for record in records:
        reference = str(record["reference"])
        hypothesis = str(record["hypothesis"])
        chars_a = [x for x in reference.lower() if not x.isspace()]
        chars_b = [x for x in hypothesis.lower() if not x.isspace()]
        words_a = re.findall(r"\S+", reference.casefold())
        words_b = re.findall(r"\S+", hypothesis.casefold())
        cer_errors += edit_distance(chars_a, chars_b)
        wer_errors += edit_distance(words_a, words_b)
        cer_refs += len(chars_a)
        wer_refs += len(words_a)
        if "reference_language" in record and "detected_language" in record:
            lid_total += 1
            lid_correct += record["reference_language"] == record["detected_language"]
    return {"records": len(records), "cer": round(cer_errors / cer_refs, 4) if cer_refs else None,
            "wer": round(wer_errors / wer_refs, 4) if wer_refs else None,
            "language_accuracy": round(lid_correct / lid_total, 4) if lid_total else None,
            "reference_characters": cer_refs, "reference_words": wer_refs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise SystemExit("Expected a list of reference/hypothesis objects")
    print(json.dumps(score(data), indent=2))


if __name__ == "__main__":
    main()

"""Frozen Fixed195 human-reference text gate over 130 independent recordings.

Uses existing single-speaker gold references; never treats unlabelled
multi-speaker transcripts as ground truth. Validates original audio hashes,
full per-case results and code/model fingerprint before scoring. Report
contains aggregate counts only; does not expose source words or hypotheses.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from phase07.quality import char_tokens, mixed_tokens, score_pair
from tools.fixed195_content_review import (
    LANGUAGE_ORDER, fingerprint, validate_singles,
)
from tools.replay_transcript_score import alignment_error_counts

SCHEMA = "meeting-phase1-fixed195-text-gate-v1"


def score_frozen_singles(
    root: Path, results_path: Path,
    *, model: str = "large-v3", device: str = "cuda",
    precision: str = "float16",
    max_overall_cer: float = 0.15,
    max_overall_deletion_rate: float = 0.05,
    max_language_cer: float = 0.25,
) -> dict:
    if any(type(x) not in (int, float) or not 0 <= x <= 1 for x in (
        max_overall_cer, max_overall_deletion_rate, max_language_cer,
    )):
        raise ValueError("Invalid content quality threshold")
    cases = validate_singles(root)
    if len(cases) != 130:
        raise ValueError("Exactly 130 pinned human-reference cases required")
    state = json.loads(results_path.read_text(encoding="utf-8"))
    expected = fingerprint(cases, model, device, precision, "singles")
    if (state.get("schema") != "meeting-fixed195-content-review-v1"
            or state.get("fingerprint") != expected
            or state.get("mode") != "singles"
            or state.get("model") != model):
        raise ValueError("ASR result code/audio/model fingerprint mismatch")
    results = state.get("single_results")
    if not isinstance(results, dict) or set(results) != {
        item["id"] for item in cases
    }:
        raise ValueError("Missing, duplicated or extra frozen ASR results")

    grouped = defaultdict(lambda: {
        "samples": 0, "reference_characters": 0,
        "character_edits": 0, "character_deletions": 0,
        "token_reference": 0, "token_deletions": 0,
        "unscored_alignment": 0, "empty_predictions": 0,
        "review_flagged": 0,
    })
    for item in cases:
        observed = results[item["id"]]
        if (not isinstance(observed, dict) or observed.get("error")
                or observed.get("reference") != item["reference"]
                or observed.get("language") != item["language"]
                or not isinstance(observed.get("hypothesis"), str)):
            raise ValueError(f"Unverifiable model result for {item['id']}")
        reference, hypothesis = item["reference"], observed["hypothesis"]
        original = score_pair(reference, hypothesis,
                              language=item["language"])
        if (observed.get("character_edits") != original["character_edits"]
                or observed.get("reference_characters")
                != original["reference_characters"]):
            raise ValueError(f"Stored CER differs from reference for {item['id']}")
        characters = alignment_error_counts(
            char_tokens(reference), char_tokens(hypothesis),
        )
        if characters is not None and characters["edits"] != original["character_edits"]:
            raise ValueError(f"Character edit attribution inconsistent: {item['id']}")
        tokens = None
        # Thai is excluded from unsupported word/token segmentation.
        if item["language"] != "th":
            ref_tokens = mixed_tokens(reference)
            hyp_tokens = mixed_tokens(hypothesis)
            tokens = alignment_error_counts(ref_tokens, hyp_tokens)
        summary = grouped[item["language"]]
        summary["samples"] += 1
        summary["reference_characters"] += original["reference_characters"]
        summary["character_edits"] += original["character_edits"]
        if characters is None:
            summary["unscored_alignment"] += 1
        else:
            summary["character_deletions"] += characters["deletions"]
        if tokens is not None:
            summary["token_reference"] += len(ref_tokens)
            summary["token_deletions"] += tokens["deletions"]
        if not hypothesis.strip():
            summary["empty_predictions"] += 1
        if observed.get("review_flags"):
            summary["review_flagged"] += 1

    per_language = {}
    total = defaultdict(int)
    for language in LANGUAGE_ORDER:
        row = grouped[language]
        if not row["samples"] or not row["reference_characters"]:
            raise ValueError(f"Missing scoreable content for {language}")
        for field, value in row.items():
            total[field] += value
        cer = row["character_edits"] / row["reference_characters"]
        deletion_rate = (
            row["character_deletions"] / row["reference_characters"]
        )
        per_language[language] = {
            **row,
            "cer": round(cer, 5),
            "character_deletion_rate": round(deletion_rate, 5),
            "max_cer_pass": cer <= max_language_cer,
        }
    overall_cer = total["character_edits"] / total["reference_characters"]
    overall_deletion = (
        total["character_deletions"] / total["reference_characters"]
    )
    gates = {
        "exactly_130_frozen_samples": total["samples"] == 130,
        "all_alignment_counts_evaluable": total["unscored_alignment"] == 0,
        "no_empty_model_predictions": total["empty_predictions"] == 0,
        "weighted_character_error_below_limit": overall_cer <= max_overall_cer,
        "weighted_character_deletion_below_limit": (
            overall_deletion <= max_overall_deletion_rate
        ),
        "each_language_cer_below_limit": all(
            per_language[language]["max_cer_pass"]
            for language in LANGUAGE_ORDER
        ),
    }
    return {
        "schema": SCHEMA,
        "pass": all(gates.values()),
        "gates": gates,
        "model": model, "device": device, "precision": precision,
        "model_and_code_fingerprint": expected,
        "asr_results_sha256": sha256(results_path.read_bytes()).hexdigest(),
        "reference_case_count": total["samples"],
        "overall": {
            "cer": round(overall_cer, 5),
            "character_deletion_rate": round(overall_deletion, 5),
            "review_flagged": total["review_flagged"],
            "empty_predictions": total["empty_predictions"],
        },
        "per_language": dict(per_language),
        "thresholds": {
            "max_overall_cer": max_overall_cer,
            "max_overall_deletion_rate": max_overall_deletion_rate,
            "max_language_cer": max_language_cer,
        },
        "note": (
            "Referenced SINGLE-speaker ASR content only; does not certify "
            "live final-caption delivery, multi-speaker accuracy, overlap, "
            "WER for Thai, or the full set of desired languages. "
            "Counts come from one minimum-edit alignment per sample."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()
    root = args.root.resolve()
    private = Path("benchmark/fixed195").resolve()
    report_path = (args.results or (root / "content_review_singles.json")).resolve()
    target = args.output.resolve()
    if (not root.is_relative_to(private)
            or not report_path.is_relative_to(root)
            or not target.is_relative_to(root)
            or not report_path.is_file()
            or target.exists() or target.suffix.lower() != ".json"):
        parser.error("Use pinned existing private Fixed195 results and a NEW private report")
    result = score_frozen_singles(
        root, report_path, model=args.model,
        device=args.device, precision=args.compute_type,
    )
    with target.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

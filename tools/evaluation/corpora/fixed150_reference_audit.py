"""Audit reference coverage in the frozen 100-case multilingual benchmark.

Flags candidates for MANUAL listening review; no model predictions are rerun,
and flagged examples are never silently deleted or relabeled. Report raw
metrics separately from sensitivity calculations that exclude flagged cases.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runtime_quality.quality import char_tokens
from tools.evaluation.corpora.fixed150_prepare import atomic_json, verify_lock

FOCUS_15 = ("en", "zh", "es", "fr", "de", "ja", "pt", "ru", "ko",
            "hi", "ar", "it", "id", "tr", "vi")


def source_flags(duration_s: float, reference: str,
                 reference_chars: int | None = None) -> list[str]:
    """Low reference-text coverage is a review trigger, not proof of bad labels."""
    count = len(char_tokens(reference))
    if reference_chars is not None and count != reference_chars:
        raise ValueError("Reference normalization mismatch; check manifest and report")
    issues = []
    if duration_s >= 15 and count / duration_s < 2:
        issues.append("LOW_REFERENCE_CHARACTER_DENSITY")
    if duration_s >= 30 and count < 65:
        issues.append("LONG_AUDIO_SHORT_REFERENCE")
    return issues


def evaluate(lock: dict, benchmark: dict, languages: tuple[str, ...],
             ablation: dict | None = None) -> dict:
    sources = [c for c in lock["cases"] if c["kind"] == "single"]
    if len(sources) != 100 or not benchmark.get("infer") or benchmark.get("samples") != 100:
        raise ValueError("Require immutable 100-case inference and lock")
    rows = benchmark["engines"]["whisper"]["samples"]
    if len(rows) != 100:
        raise ValueError("Missing sample scores")
    extras = {c["id"]: c for c in (ablation or {}).get("cases", [])}
    audit = []
    for i, (case, row) in enumerate(zip(sources, rows)):
        if row.get("index") != i or row.get("error"):
            raise ValueError(f"Failed or misaligned inference row {i}")
        flags = source_flags(case["duration_s"], case["reference"],
                             row.get("reference_characters"))
        existing = extras.get(case["id"])
        model_length = None
        if existing:
            prediction = existing.get("pipeline_auto", {}).get("text")
            if isinstance(prediction, str):
                model_length = len(char_tokens(prediction))
                count = row["reference_characters"]
                if count and model_length / count >= 1.8 and row["cer"] >= .50:
                    flags.append("MODEL_MUCH_LONGER_THAN_REFERENCE")
        audit.append({
            "id": case["id"], "language": case["language"],
            "duration_s": case["duration_s"],
            "reference_characters": row["reference_characters"],
            "reference_density_chars_s": round(
                row["reference_characters"] / case["duration_s"], 3),
            "predicted_characters_available": model_length,
            "cer": row["cer"],
            "flags": flags,
            "review_required": bool(flags),
        })
    supported = set(languages)
    by_lang = defaultdict(list)
    for source, line, row in zip(sources, audit, rows):
        if source["language"] in supported:
            by_lang[source["language"]].append((line, row))

    def rate(items: list[tuple[dict, dict]]) -> float | None:
        den = sum(row["reference_characters"] for _, row in items)
        return (sum(row["character_edits"] for _, row in items) / den) if den else None

    report = []
    for lang in languages:
        items = by_lang.get(lang, [])
        clean = [(entry, row) for entry, row in items if not entry["review_required"]]
        report.append({
            "language": lang, "samples": len(items),
            "flagged_for_review": len(items) - len(clean),
            "raw_cer": rate(items),
            "sensitivity_cer_excluding_flagged": rate(clean),
            "sensitivity_n": len(clean),
        })
    all_items = [item for items in by_lang.values() for item in items]
    clear = [(entry, row) for entry, row in all_items if not entry["review_required"]]
    return {
        "schema": "meeting-fixed150-reference-audit-v1",
        "selected_languages": list(languages),
        "selected_samples": len(all_items),
        "raw_selected_cer": rate(all_items),
        "flagged_samples": sum(entry["review_required"] for entry, _ in all_items),
        "sensitivity_cer_excluding_flagged": rate(clear),
        "sensitivity_samples": len(clear),
        "per_language": report,
        "review_candidates": [entry for entry in audit if entry["review_required"]],
        "all_100_original_cases_retained": True,
        "release_certified": False,
        "interpretation": [
            "Flags are statistical heuristics, NOT evidence that ground truth is wrong.",
            "Listen to each flagged WAV and verify the transcription before making corrections.",
            "Raw selected CER includes all selected source cases without modification.",
            "Sensitivity CER is only a hypothetical comparison, never a substitute for audited accuracy.",
            "Do not overwrite the source dataset's lock, raw manifest or initial GPU results.",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    p.add_argument("--languages", nargs="+", default=list(FOCUS_15))
    args = p.parse_args()
    if len(set(args.languages)) != len(args.languages):
        p.error("Duplicate language code")
    root = args.root
    lock = verify_lock(root)
    benchmark = json.loads((root / "single_asr.json").read_text(encoding="utf-8"))
    maybe = root / "local_failure_ablation.json"
    ablation = json.loads(maybe.read_text(encoding="utf-8")) if maybe.is_file() else None
    result = evaluate(lock, benchmark, tuple(args.languages), ablation)
    target = root / "reference_audit.json"
    atomic_json(target, result)
    print(f"15-language (or selected language) original CER: {result['raw_selected_cer']:.2%}")
    print(f"Needs manual label/audio review: {result['flagged_samples']} / "
          f"{result['selected_samples']}")
    print("  Language | n | flagged | original CER | sensitivity only")
    for row in result["per_language"]:
        raw = "N/A" if row["raw_cer"] is None else f"{row['raw_cer']:.2%}"
        sensitive = ("N/A" if row["sensitivity_cer_excluding_flagged"] is None
                     else f"{row['sensitivity_cer_excluding_flagged']:.2%}")
        print(f"  {row['language']:<8} | {row['samples']:2} | "
              f"{row['flagged_for_review']:2} | {raw:>10} | {sensitive:>10}")
    print("Flagged source cases (listen and review, do not auto-delete):")
    for entry in result["review_candidates"]:
        print(" ", entry["id"], entry["language"],
              f"{entry['duration_s']}s", ", ".join(entry["flags"]))
    print("Audit JSON:", target)
    print("Neither reference text nor model output was automatically changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

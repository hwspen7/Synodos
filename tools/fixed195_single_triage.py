"""Triage per-clip errors in a completed Fixed195 single-speaker run.

This script reads local reference/hypothesis comparisons only, performs no
new inference, and never uploads raw transcripts. Ranked cases are for
investigation, not cherry-picking cases out of the official 130-clip metric.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fixed150_prepare import atomic_json
from tools.fixed195_prepare import LANGUAGES, singles_for

LANGUAGE_ORDER = [code for code, _ in LANGUAGES]
EXPECTED_SCHEMA = "meeting-fixed195-content-review-v1"
EXPECTED_SINGLE_COUNT = 130


def case_entry(row: dict) -> dict:
    return {
        "id": row["id"],
        "language": row["language"],
        "cer_percent": round(100 * row["character_edits"]
                             / row["reference_characters"], 3),
        "edits": row["character_edits"],
        "reference_chars": row["reference_characters"],
        "predicted_language": row.get("predicted_language"),
        "duration_s": row.get("duration_s"),
        "latency_ms": row.get("latency_ms"),
        "source_group": row["source_group"],
        "review_flags": row.get("review_flags") or [],
    }


def group_summary(rows: list[dict]) -> dict:
    ref = sum(x["reference_characters"] for x in rows)
    errors = sum(x["character_edits"] for x in rows)
    return {
        "samples": len(rows),
        "reference_chars": ref,
        "edits": errors,
        "cer_percent": round(100 * errors / ref, 3) if ref else None,
    }


def triage(cases: list[dict], results: dict, *, top: int = 4,
           require_complete: bool = True) -> dict:
    if not (1 <= top <= 30):
        raise ValueError("--top must be 1..30")
    manifest = {x["id"]: x for x in cases}
    if len(manifest) != len(cases):
        raise ValueError("Duplicate IDs in single manifest")
    if (require_complete and len(cases) != EXPECTED_SINGLE_COUNT):
        raise ValueError(f"Expected {EXPECTED_SINGLE_COUNT} singles")
    if set(results) - set(manifest):
        raise ValueError("Unknown result IDs")
    complete = []
    failures = []
    for sample in cases:
        result = results.get(sample["id"])
        if result is None:
            continue
        if result.get("error"):
            failures.append({"id": sample["id"], "error": result["error"]})
            continue
        if result.get("language") != sample["language"]:
            raise ValueError(f"Case language changed: {sample['id']}")
        if not isinstance(result.get("character_edits"), int):
            raise ValueError(f"Missing integer character edits: {sample['id']}")
        if not isinstance(result.get("reference_characters"), int):
            raise ValueError(f"Missing integer reference length: {sample['id']}")
        if result["character_edits"] < 0 or result["reference_characters"] <= 0:
            raise ValueError(f"Invalid CER counts: {sample['id']}")
        if not isinstance(result.get("reference"), str):
            raise ValueError(f"Missing transcription reference: {sample['id']}")
        if result["reference"] != sample["reference"]:
            raise ValueError(f"Reference text changed: {sample['id']}")
        if not isinstance(result.get("hypothesis"), str):
            raise ValueError(f"Missing recognized transcript: {sample['id']}")
        complete.append({
            **result,
            "source_group": ("old" if sample.get("reused_fixed150_id")
                             else "new"),
        })
    if require_complete and (len(complete) != EXPECTED_SINGLE_COUNT or failures):
        raise ValueError(f"Expected all 130 evaluated; got {len(complete)} with "
                         f"{len(failures)} failures")

    grouped = defaultdict(list)
    for item in complete:
        grouped[item["language"]].append(item)
    languages = {}
    for language in LANGUAGE_ORDER:
        group = grouped[language]
        old = [x for x in group if x["source_group"] == "old"]
        new = [x for x in group if x["source_group"] == "new"]
        worst_cer = sorted(
            group,
            key=lambda x: (-x["character_edits"] / x["reference_characters"],
                           -x["character_edits"], x["id"]),
        )[:top]
        worst_edits = sorted(
            group,
            key=lambda x: (-x["character_edits"],
                           -x["character_edits"] / x["reference_characters"],
                           x["id"]),
        )[:top]
        total_edits = sum(x["character_edits"] for x in group)
        top_three_edits = sum(x["character_edits"] for x in worst_edits[:3])
        language_mismatches = [
            {"id": x["id"], "detected": x["predicted_language"]}
            for x in group
            if x.get("predicted_language") not in
            (None, "", "und", "mul", language)
        ]
        unknown_or_mixed = [
            {"id": x["id"], "detected": x.get("predicted_language")}
            for x in group
            if x.get("predicted_language") in ("und", "mul", None, "")
        ]
        languages[language] = {
            "summary": group_summary(group),
            "old": group_summary(old),
            "new": group_summary(new),
            "top_by_cer": [case_entry(x) for x in worst_cer],
            "top_by_edits": [case_entry(x) for x in worst_edits],
            "top_three_edit_share_percent": (
                round(100 * top_three_edits / total_edits, 2)
                if total_edits else 0.0
            ),
            "detected_language_mismatch": language_mismatches,
            "undetermined_or_multilingual_labels": unknown_or_mixed,
            "cases_with_review_flags": [
                x["id"] for x in group if x.get("review_flags")
            ],
        }
        if require_complete and len(group) != singles_for(language):
            raise ValueError(f"Wrong evaluated count for {language}: {len(group)}")

    return {
        "schema": "meeting-fixed195-single-triage-v1",
        "overall": group_summary(complete),
        "completed": len(complete),
        "requested": len(cases),
        "failed": failures,
        "not_run": len(cases) - len(complete) - len(failures),
        "per_language": languages,
        "notes": [
            "Retain all examples in official metrics; outlier review is diagnostic.",
            "High CER could reflect acoustic segmentation, language recognition, "
            "reference content or other processing faults; no cause is inferred.",
            "Language labels 'mul' and 'und' are not treated as confident "
            "wrong-language predictions.",
            "Old and new audio sources differ; comparing their CER does not "
            "measure performance improvement over time.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=LANGUAGE_ORDER)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--show-text", action="store_true",
                        help="Print local reference/hypothesis pairs for top-ranked clips")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    source = args.results or args.root / "content_review_singles.json"
    manifest = args.root / "manifest_single.jsonl"
    if not manifest.exists() or not source.exists():
        parser.error("Required manifest/results missing; run single inference first")
    state = json.loads(source.read_text(encoding="utf-8"))
    if state.get("schema") != EXPECTED_SCHEMA or state.get("mode") != "singles":
        parser.error("Expected Fixed195 completed single-speaker evaluation report")
    cases = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    report = triage(cases, state.get("single_results", {}), top=args.top,
                    require_complete=not args.allow_partial)
    path = args.output or args.root / "content_review_singles_triage.json"
    atomic_json(path, report)
    languages = [args.language] if args.language else LANGUAGE_ORDER
    for language in languages:
        data = report["per_language"][language]
        total, old, new = (data[key]["cer_percent"]
                           for key in ("summary", "old", "new"))
        print(f"{language:>2}: {total}% CER (old={old}%, new={new}%) "
              f"top3_edit_share={data['top_three_edit_share_percent']}%")
        for item in data["top_by_edits"]:
            print(f"  {item['id']} CER={item['cer_percent']}% "
                  f"edits={item['edits']}/{item['reference_chars']} "
                  f"source={item['source_group']} "
                  f"detected={item['predicted_language']}")
            if args.show_text:
                row = state["single_results"][item["id"]]
                print(f"    REFERENCE: {row['reference']}")
                print(f"    HYPOTHESIS: {row['hypothesis']}")
    print(f"Local diagnostic report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

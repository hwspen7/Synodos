"""Phase 1 approved 10-language gate over the immutable Fixed195 archive.

This keeps the original 130 audio samples and historical twelve-language
results unchanged. Recompute acceptance from the complete original source
using 110 supported cases only; NEVER mislabel subset selection as a model
accuracy improvement or discard data from the frozen source archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech.supported_languages import (
    PHASE1_LANGUAGE_CODES, PHASE1_LANGUAGE_SET,
)
from tools.validation.fixed195_text_gate import score_frozen_singles


SCHEMA = "meeting-phase1-ten-language-text-gate-v1"
EXCLUDED_ARCHIVE_LANGUAGES = frozenset(("bn", "pt"))


def resolve_original_asr_results(path: Path) -> Path:
    """Recover the paired raw report when given a generated summary JSON.

    Fixed195 writes both 'NAME.json' (130 case-level results) and
    'NAME_summary.json' (aggregate-only). The summary file typically has
    the newest modification time and is easily selected by a broad glob.
    Only the exact sibling file is considered; the original full report
    must still pass the unchanged fingerprint, audio and reference checks.
    """
    if not path.name.endswith("_summary.json"):
        return path

    summary = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(summary, dict)
            or summary.get("schema")
            != "meeting-fixed195-content-review-v1-summary"
            or summary.get("mode") != "singles"):
        raise ValueError(
            "The supplied _summary.json is not a valid Fixed195 singles "
            "summary. Pass the original 130-case ASR results JSON instead."
        )

    original = path.with_name(
        path.name[:-len("_summary.json")] + ".json"
    )
    if not original.is_file():
        raise FileNotFoundError(
            "Fixed195 summary has no matching original case-level report: "
            f"{original}. Re-run inference with a new --output name if "
            "the original report is unavailable."
        )
    raw = json.loads(original.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict)
            or raw.get("schema") != "meeting-fixed195-content-review-v1"
            or raw.get("mode") != "singles"
            or not isinstance(raw.get("single_results"), dict)):
        raise ValueError(
            "The paired report is not a complete Fixed195 singles result "
            "file; do not score aggregate-only summaries."
        )
    print(f"Using original ASR results paired with summary: {original}")
    return original


def evaluate_ten_language_selection(
    original_report: dict, *,
    max_overall_cer: float = 0.15,
    max_overall_deletion_rate: float = 0.05,
    max_language_cer: float = 0.25,
) -> dict:
    if (not isinstance(original_report, dict)
            or original_report.get("schema")
            != "meeting-phase1-fixed195-text-gate-v1"
            or original_report.get("reference_case_count") != 130
            or any(type(v) not in (int, float) or not 0 <= v <= 1
                   for v in (
                       max_overall_cer, max_overall_deletion_rate,
                       max_language_cer,
                   ))):
        raise ValueError("Complete original 130-case reference audit required")
    all_groups = original_report.get("per_language")
    if not isinstance(all_groups, dict):
        raise ValueError("Full archive language detail is required")
    if set(all_groups) != PHASE1_LANGUAGE_SET | EXCLUDED_ARCHIVE_LANGUAGES:
        raise ValueError("Frozen twelve-language archive has unexpected selection")
    selected = {language: all_groups[language] for language in PHASE1_LANGUAGE_CODES}
    if (sum(int(x["samples"]) for x in all_groups.values()) != 130
            or sum(int(x["samples"]) for x in selected.values()) != 110
            or selected["en"]["samples"] != 20
            or any(row["samples"] != 10
                   for language, row in selected.items() if language != "en")):
        raise ValueError("Incorrect frozen selection: expected 110 source clips")
    if any(
        type(row[key]) is not int or row[key] < 0
        for row in selected.values()
        for key in (
            "samples", "reference_characters", "character_edits",
            "character_deletions", "unscored_alignment",
            "empty_predictions",
        )
    ):
        raise ValueError("Invalid source scoring counts")
    total_reference = sum(row["reference_characters"] for row in selected.values())
    total_edits = sum(row["character_edits"] for row in selected.values())
    total_deleted = sum(row["character_deletions"] for row in selected.values())
    if not total_reference or total_deleted > total_edits:
        raise ValueError("Invalid source aggregate edit counts")
    cer = total_edits / total_reference
    deletion_rate = total_deleted / total_reference
    per_language = {
        lang: {
            "samples": row["samples"],
            "reference_characters": row["reference_characters"],
            "character_edits": row["character_edits"],
            "character_deletions": row["character_deletions"],
            "cer": round(
                row["character_edits"] / row["reference_characters"], 5,
            ),
            "deletion_rate": round(
                row["character_deletions"] / row["reference_characters"], 5,
            ),
            "pass": (
                row["character_edits"] / row["reference_characters"]
                <= max_language_cer
            ),
        }
        for lang, row in selected.items()
        if row["reference_characters"] > 0
    }
    gates = {
        "exactly_ten_supported_languages": (
            len(selected) == 10 and len(per_language) == 10
        ),
        "exactly_110_frozen_reference_clips": (
            sum(row["samples"] for row in selected.values()) == 110
        ),
        "all_selected_alignments_evaluable": all(
            row["unscored_alignment"] == 0 for row in selected.values()
        ),
        "no_empty_predictions": all(
            row["empty_predictions"] == 0 for row in selected.values()
        ),
        "weighted_cer_within_limit": cer <= max_overall_cer,
        "weighted_deletion_within_limit": (
            deletion_rate <= max_overall_deletion_rate
        ),
        "each_supported_language_within_limit": (
            len(per_language) == 10 and
            all(row["pass"] for row in per_language.values())
        ),
    }
    return {
        "schema": SCHEMA, "pass": all(gates.values()),
        "gates": gates,
        "supported_language_codes": list(PHASE1_LANGUAGE_CODES),
        "excluded_from_phase1": sorted(EXCLUDED_ARCHIVE_LANGUAGES),
        "archive_reference_case_count": 130,
        "selected_reference_case_count": 110,
        "overall": {
            "reference_characters": total_reference,
            "character_edits": total_edits,
            "character_deletions": total_deleted,
            "cer": round(cer, 5),
            "character_deletion_rate": round(deletion_rate, 5),
        },
        "per_language": per_language,
        "source_model_and_code_fingerprint": original_report.get(
            "model_and_code_fingerprint"
        ),
        "source_asr_results_sha256": original_report.get(
            "asr_results_sha256"
        ),
        "thresholds": {
            "max_overall_cer": max_overall_cer,
            "max_overall_deletion_rate": max_overall_deletion_rate,
            "max_language_cer": max_language_cer,
        },
        "limitations": (
            "This is a pre-specified subset of existing single-speaker "
            "reference-backed audio, not a new Whisper inference improvement "
            "or an end-to-end live meeting transcript test. The original "
            "twelve-language archive remains unchanged and reproducible."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()
    frozen = Path("benchmark/fixed195").resolve()
    root, results, output = (
        args.root.resolve(), args.results.resolve(), args.output.resolve(),
    )
    if (not root.is_relative_to(frozen)
            or not results.is_relative_to(root)
            or not results.is_file()
            or not output.is_relative_to(root)
            or output.suffix.lower() != ".json"
            or output.exists()):
        parser.error("Use the frozen private archive and a new JSON output")
    results = resolve_original_asr_results(results)
    archive = score_frozen_singles(
        root, results, model=args.model, device=args.device,
        precision=args.compute_type,
    )
    selection = evaluate_ten_language_selection(archive)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2)
        stream.write("\n")
    print(json.dumps(selection, indent=2))
    return 0 if selection["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

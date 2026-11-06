"""Strict artifact-backed Phase 1/2 release contract before Phase 3.

Every artifact is produced by a different frozen test path. Green GitHub CI
is necessary but not sufficient. The operator must independently review
the mixed-audio human transcript before claiming text accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "meeting-phase12-third-stage-readiness-v1"
REQUIRED_CI = {
    "Backend syntax and unit tests",
    "Frontend type-check and build",
    "PostgreSQL integration tests",
    "Backup integrity, restore safety, and load-test guards",
    "powershell-parser",
}
REQUIRED_ABLATIONS = {
    "content_first",
    "whisper_window_vad_bypass",
    "bounded_vad_gap_recovery",
    "preserve_verified_unaligned",
    "word_timestamp_bounds",
}


def assess(
    language: dict, human: dict, acoustic: dict, comparison: dict,
    qwen_shadow: dict, identity: dict, ci: dict, ablation: dict,
    *, source_commit: str,
) -> dict:
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("A pinned 40-character Git commit SHA is required")
    if human.get("schema") != "meeting-phase1-human-text-acceptance-v1":
        raise ValueError("Unknown human-reference acceptance")
    if acoustic.get("schema") != "meeting-ami-independent-nemotron-live-score-v1":
        raise ValueError("Unknown independent acoustic evidence")
    if comparison.get("schema") != "meeting-phase1b-paired-comparison-v1":
        raise ValueError("Unknown paired-model comparison")
    if identity.get("schema") != "meeting-phase2-cross-session-acceptance-v1":
        raise ValueError("Unknown cross-session identity evidence")
    if qwen_shadow.get("schema") != "meeting-phase1b-qwen-asr-shadow-v1":
        raise ValueError("Unknown Qwen shadow experiment")
    if ci.get("schema") != "meeting-phase1-github-actions-diagnostic-v1":
        raise ValueError("Unknown GitHub checks diagnostic")
    if ablation.get("schema") != "meeting-phase1a-five-candidate-ablation-v1":
        raise ValueError("Unknown Phase 1A experiment inventory")

    lg = language.get("gates") or {}
    real = human.get("gates") or {}
    overall = language.get("overall") or {}
    supported = language.get("supported_language_codes") or []
    required_ten = {"en", "zh", "fr", "es", "ar",
                    "id", "ja", "ru", "de", "hi"}
    acoustic_metrics = acoustic.get("independent_acoustic_metrics") or {}
    checked_runs = ci.get("checks") or []
    completed_checks = {
        item.get("name") for item in checked_runs
        if (isinstance(item, dict)
            and item.get("conclusion") == "success"
            and item.get("status") == "completed"
            and item.get("github_actions_job") is True
            and int(item.get("executed_steps") or 0) > 0)
    }
    ablation_modes = {
        item.get("variant") for item in ablation.get("candidate_trials") or []
        if item.get("independent_der_path")
    }
    asr_cer = real.get("character_error_at_most_threshold") is True
    gates = {
        "ten_language_frozen_cer": (
            language.get("schema") == "meeting-phase1-ten-language-text-gate-v1"
            and language.get("pass") is True
            and set(supported) == required_ten
            and language.get("selected_reference_case_count") == 110
            and all(value is True for value in lg.values())
            and len(lg) == 7
            and type(overall.get("cer")) in (int, float)
            and 0 <= overall["cer"] < 0.10
        ),
        "reviewed_real_meeting_text": (
            human.get("pass") is True
            and real.get("human_reference_confirmed") is True
            and real.get("confirmed_text_not_silently_revised") is True
            and real.get("confirmed_segments_never_disappeared") is True
            and real.get("reference_deletion_at_most_threshold") is True
            and asr_cer
        ),
        "same_audio_der_under_20_percent": (
            acoustic.get("meeting") == "TS3003b"
            and acoustic_metrics.get("der") is not None
            and type(acoustic_metrics.get("der")) in (int, float)
            and 0 <= acoustic_metrics["der"] < 0.20
            and acoustic.get("under_20pct_standard_der") is True
            and acoustic.get("source_audio_sha256")
                == human.get("source_audio_sha256")
        ),
        "five_asr_ablation_traces": (
            ablation_modes == REQUIRED_ABLATIONS
            and ablation.get("source_commit") == source_commit
        ),
        "three_model_paired_ten_meeting_comparison": (
            comparison.get("complete_heldout_cohort") is True
            and comparison.get("paired_meeting_count") == 10
            and comparison.get("acoustic_goal_met") is True
            and comparison.get("may_auto_promote_to_live") is False
        ),
        "qwen_non_mutating_shadow_executed": (
            qwen_shadow.get("infer_enabled") is True
            and qwen_shadow.get("reference_words_used_to_select_windows") is False
            and qwen_shadow.get("auto_replaced_live_text") is False
            and qwen_shadow.get("detected_candidate_windows", 0) > 0
            and len(qwen_shadow.get("candidate_transcripts") or []) > 0
        ),
        "phase2_disjoint_speaker_holdout": (
            identity.get("holdout_goal_met") is True
            and identity.get("dev_pairs", 0) >= 40
            and identity.get("holdout_pairs", 0) >= 40
            and identity.get("may_auto_update_production_threshold") is False
        ),
        "hosted_checks_for_exact_commit": (
            ci.get("commit_sha") == source_commit
            and ci.get("all_required_checks_green") is True
            and REQUIRED_CI.issubset(completed_checks)
        ),
    }
    return {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "pass": all(gates.values()),
        "gates": gates,
        "passed_count": sum(gates.values()),
        "total_gate_count": len(gates),
        "next_stage": "phase3" if all(gates.values()) else "phase1_2_blocked",
        "automatic_merge_or_tag": False,
        "note": (
            "This is a reproducible engineering release contract, not a "
            "commercial benchmark or a promise of universal accuracy. "
            "No artifact may be omitted. A human must independently "
            "review acoustic overlap and the real transcript."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("language", "human", "acoustic", "comparison",
                "qwen-shadow", "identity", "ci", "ablation"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path("benchmark/fixed195").resolve()
    names = ("language", "human", "acoustic", "comparison",
             "qwen_shadow", "identity", "ci", "ablation")
    paths = [getattr(args, key).resolve(strict=True) for key in names]
    target = args.output.resolve()
    if (any(not path.is_relative_to(root) for path in paths)
            or not target.is_relative_to(root)
            or target.suffix.lower() != ".json" or target.exists()):
        parser.error("Use private immutable Fixed195 inputs and a NEW JSON output")
    result = assess(
        *(json.loads(path.read_text(encoding="utf-8")) for path in paths),
        source_commit=args.source_commit,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

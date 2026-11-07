"""Explain the completed system-wide Fixed150 ablation without running a model.

This summarizes existing local numeric measurements only. It does not mutate
original audio, references or ASR reports, and never treats a suspected failure
stage as proven causal without a corresponding controlled stage ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect(report: dict, threshold_pp: float = 5.0) -> dict:
    if report.get("schema") != "meeting-fixed150-system-ablation-v1":
        raise ValueError("Unrecognized system ablation schema")
    cases = report.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Missing measured cases")
    rows = []
    for cid, case in cases.items():
        modes = ("production_windowed", "production_whole_clip", "direct_whisper")
        if case.get("id") != cid or not all(
            isinstance(case.get(m), dict) and isinstance(case[m].get("cer"), (int, float))
            for m in modes
        ):
            raise ValueError(f"Incomplete numeric metrics: {cid}")
        window, whole, raw = (case[m]["cer"] for m in modes)
        geometry = case.get("acoustic_geometry", {})
        flags = []
        if 100 * (window - whole) >= threshold_pp:
            flags.append("WINDOW_ROUTING_CANDIDATE")
        if 100 * (whole - raw) >= threshold_pp:
            flags.append("POSTPROCESS_OR_DECODING_DIFFERENCE_CANDIDATE")
        if case.get("reference_review_flags"):
            flags.append("REFERENCE_REVIEW_REQUIRED")
        if geometry.get("window_boundary_inside_detected_speech", 0) > 0:
            flags.append("WINDOW_SEAM_WITHIN_DETECTED_SPEECH")
        rows.append({
            "id": cid, "language": case["language"],
            "duration_s": case["duration_s"],
            "asr_windows": geometry.get("asr_windows"),
            "speech_cut_seams": geometry.get("window_boundary_inside_detected_speech", 0),
            "cer_window_percent": round(window * 100, 2),
            "cer_global_percent": round(whole * 100, 2),
            "cer_direct_percent": round(raw * 100, 2),
            "window_minus_global_pp": round(100 * (window - whole), 2),
            "global_minus_direct_pp": round(100 * (whole - raw), 2),
            "window_minus_direct_pp": round(100 * (window - raw), 2),
            "flags": flags,
        })
    ranked = sorted(rows, key=lambda r: r["window_minus_direct_pp"], reverse=True)
    impacted = [r for r in rows if r["window_minus_direct_pp"] >= threshold_pp]
    categories = Counter(label for row in rows for label in row["flags"])
    return {
        "schema": "meeting-fixed150-system-triage-v1",
        "cases": len(rows),
        "threshold_pp": threshold_pp,
        "per_language_affected": dict(sorted(Counter(
            r["language"] for r in impacted
        ).items())),
        "cases_harmed_by_windowed_vs_direct": len(impacted),
        "flagged_cause_candidates": dict(categories),
        "ranked": ranked,
        "important_notes": [
            "Stage labels are hypotheses, not isolation of a precise implementation defect.",
            "A case can worsen in one stage and improve in another.",
            "Audio clipping at a window seam is not established by a VAD seam flag alone.",
            "Leave raw reference annotations and baseline CER unchanged.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("benchmark/fixed150/system_ablation_all.json"))
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--threshold-pp", type=float, default=5.0)
    args = parser.parse_args()
    if args.top < 1 or args.threshold_pp <= 0:
        parser.error("--top and --threshold-pp must be positive")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    triage = inspect(report, args.threshold_pp)
    print("Original clips assessed:", triage["cases"])
    print("Windowed >=5pp worse than direct:",
          triage["cases_harmed_by_windowed_vs_direct"])
    print("By language:", triage["per_language_affected"])
    print("Candidates:", triage["flagged_cause_candidates"])
    print()
    print("Worst paired regressions (CER %, difference in percentage points):")
    print("ID            Lang  Seconds  Wins  Speech seams   Window  Global   Direct  W-G   G-R")
    for case in triage["ranked"][:args.top]:
        print(
            f"{case['id']:<13} {case['language']:<5} {case['duration_s']:7.1f} "
            f"{str(case['asr_windows']):>5} {case['speech_cut_seams']:13} "
            f"{case['cer_window_percent']:8.1f} {case['cer_global_percent']:7.1f} "
            f"{case['cer_direct_percent']:8.1f} "
            f"{case['window_minus_global_pp']:5.1f} "
            f"{case['global_minus_direct_pp']:5.1f}"
        )
    print()
    print("Candidate flags for above cases:")
    for case in triage["ranked"][:args.top]:
        print(case["id"], ", ".join(case["flags"]) or "none")
    print("This triage reuses existing JSON and does not execute GPU inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

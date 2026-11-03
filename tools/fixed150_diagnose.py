"""Audit actual Fixed150 inference without downloading or rerunning the ASR model.

Read immutable source manifest and already-measured numerical predictions.
Does not persist raw reference or hypothesis transcripts by default.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fixed150_prepare import verify_lock


def diagnose(lock: dict, report: dict, engine: str = "whisper") -> dict:
    originals = [c for c in lock["cases"] if c["kind"] == "single"]
    if len(originals) != 100 or not report.get("infer") or report.get("samples") != 100:
        raise ValueError("Require a 100-original-sample real-inference report")
    result = report.get("engines", {}).get(engine)
    if not isinstance(result, dict) or len(result.get("samples", [])) != 100:
        raise ValueError(f"Missing 100 inference rows for {engine}")
    rows = result["samples"]
    groups = defaultdict(list)
    problem_cases = []
    wrong_languages = []
    for i, (case, metric) in enumerate(zip(originals, rows)):
        if metric.get("index") != i:
            raise ValueError(f"Unaligned report at index {i}")
        if metric.get("error"):
            problem_cases.append({
                "id": case["id"], "language": case["language"], "error": metric["error"]
            })
            continue
        if (not metric.get("reference_characters") or
            not isinstance(metric.get("cer"), (int, float)) or
            not math.isfinite(metric["cer"])):
            raise ValueError(f"Invalid CER for {case['id']}")
        groups[case["language"]].append((case, metric))
        if metric.get("expected_language") != metric.get("predicted_language"):
            wrong_languages.append({
                "id": case["id"], "expected_language": case["language"],
                "predicted_language": metric.get("predicted_language"),
                "cer_percent": round(metric["cer"] * 100, 2),
            })
        problem_cases.append({
            "id": case["id"], "language": case["language"],
            "locale": case["locale"], "scenario": case["scenario"],
            "cer_percent": round(metric["cer"] * 100, 2),
            "reference_characters": metric["reference_characters"],
            "character_edits": metric["character_edits"],
            "predicted_language": metric.get("predicted_language"),
            "latency_ms": round(float(metric["latency_ms"]), 2),
        })
    if len(problem_cases) != 100:
        raise ValueError("Report must account for all 100 cases")
    per_language = []
    for code, values in sorted(groups.items()):
        n = sum(r["reference_characters"] for _, r in values)
        per_language.append({
            "language": code, "count": len(values),
            "weighted_cer_percent": round(sum(r["character_edits"] for _, r in values) * 100 / n, 2),
            "language_accuracy_percent": round(
                sum(r.get("predicted_language") == code for _, r in values) * 100 / len(values), 1),
            "token_error_suitable_for_comparison": code != "th",
        })
    latencies = sorted(x["latency_ms"] for x in rows if isinstance(x.get("latency_ms"), (int, float)))
    report_summary = result.get("summary", {})
    return {
        "schema": "meeting-fixed150-diagnose-v1",
        "samples": 100, "failed": report_summary.get("failed", 0),
        "model_load_ms": report.get("whisper_model_load_ms"),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_ms": round(latencies[math.ceil(len(latencies) * .95) - 1], 2)
                            if latencies else None,
        "languages": per_language,
        "wrong_languages": wrong_languages,
        "worst_cases": sorted(
            problem_cases, key=lambda r: r.get("cer_percent", -1), reverse=True
        )[:20],
        "all_cases": problem_cases,
        "methodology": [
            "Raw recognized text is not in the original phase07 report.",
            "High CER requires listening to the source and comparing an opt-in transcript.",
            "Thai space-based mixed-token ER is not suitable for cross-language ranking.",
            "Each language has about four sources: per-language outliers have high variance.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--engine", default="whisper")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.top <= 100:
        parser.error("--top must be 1..100")
    lock = verify_lock(args.root)
    report = json.loads((args.root / "single_asr.json").read_text(encoding="utf-8"))
    outcome = diagnose(lock, report, args.engine)
    out = args.root / "diagnostics.json"
    out.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    with (args.root / "diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ("id", "language", "locale", "scenario", "cer_percent",
                  "reference_characters", "character_edits", "predicted_language",
                  "latency_ms")
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outcome["all_cases"])
    print(f"{outcome['samples']} cases | {outcome['failed']} inference errors")
    print(f"Whisper preload: {outcome['model_load_ms']} ms")
    print(f"Latency p50/p95: {outcome['latency_p50_ms']}/{outcome['latency_p95_ms']} ms")
    print("Language breakdown (CER, sample count):")
    for r in sorted(outcome["languages"], key=lambda v: v["weighted_cer_percent"], reverse=True):
        print(f"  {r['language']:<3} {r['weighted_cer_percent']:7.2f}%   n={r['count']}")
    print("Language-ID mismatches:")
    for item in outcome["wrong_languages"]:
        print(f"  {item['id']} {item['expected_language']} -> {item['predicted_language']}  "
              f"CER={item['cer_percent']}%")
    print(f"Highest {args.top} source-level CERs:")
    for row in outcome["worst_cases"][:args.top]:
        print(f"  {row['id']} {row['language']}  {row.get('cer_percent', 'FAILED')}%")
    print("Audit files:", out, args.root / "diagnostics.csv")
    print("No model inference or network download was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

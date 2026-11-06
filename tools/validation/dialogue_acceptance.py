"""Fail-closed acceptance gates for actual multilingual ASR benchmark reports.

Real model inference is required: simulated hypotheses cannot certify quality.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def evaluate(report: dict, languages: list[str], min_per_language: int = 5,
             max_cer: float = 0.15, max_token_error: float = 0.25,
             max_latency_p95_ms: float = 3500, min_language_accuracy: float = 0.95) -> dict:
    if report.get("infer") is not True:
        raise ValueError("Only reports generated with --infer may pass acceptance")
    if not languages or min_per_language < 1:
        raise ValueError("A nonempty language scope and labeled corpus are required")
    engines = report.get("engines") or {}
    if not engines:
        raise ValueError("Benchmark report has no engines")
    checks = {}
    for engine, payload in engines.items():
        grouped = defaultdict(list)
        for row in payload.get("samples", []):
            lang = row.get("expected_language")
            if lang in languages:
                grouped[lang].append(row)
        failures = []
        per_lang = {}
        for lang in languages:
            rows = grouped[lang]
            valid = [r for r in rows if not r.get("error")]
            if len(rows) < min_per_language or len(valid) != len(rows):
                failures.append(f"{lang}: insufficient successful real-inference samples")
                continue
            chars = sum(int(r.get("reference_characters", 0)) for r in valid)
            tokens = sum(int(r.get("reference_tokens", 0)) for r in valid)
            if chars < 1 or tokens < 1:
                failures.append(f"{lang}: missing reference content")
                continue
            cer = sum(int(r.get("character_edits", 0)) for r in valid) / chars
            ter = sum(int(r.get("token_edits", 0)) for r in valid) / tokens
            tagged = [r for r in valid if r.get("predicted_language")]
            accuracy = (sum(r["predicted_language"] == lang for r in tagged) / len(tagged)
                        if len(tagged) == len(valid) else 0.0)
            timing = sorted(float(r["latency_ms"]) for r in valid if
                            isinstance(r.get("latency_ms"), (int, float))
                            and math.isfinite(float(r["latency_ms"])) and r["latency_ms"] >= 0)
            if len(timing) != len(valid):
                failures.append(f"{lang}: incomplete inference timing data")
                continue
            p95 = timing[min(len(timing)-1, math.ceil(.95 * len(timing))-1)]
            per_lang[lang] = {"samples": len(valid), "cer": round(cer, 4),
                              "mixed_token_error": round(ter, 4),
                              "language_accuracy": round(accuracy, 4),
                              "latency_p95_ms": round(p95, 1)}
            if cer > max_cer or ter > max_token_error:
                failures.append(f"{lang}: transcription errors exceed thresholds")
            if accuracy < min_language_accuracy:
                failures.append(f"{lang}: language detection accuracy below threshold")
            if p95 > max_latency_p95_ms:
                failures.append(f"{lang}: processing latency exceeds threshold")
        checks[engine] = {"pass": not failures, "failures": failures, "per_language": per_lang}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--languages", default="en,zh,es,fr,de,ja,ko,pt,ru,it")
    parser.add_argument("--min-per-language", type=int, default=5)
    parser.add_argument("--max-cer", type=float, default=0.15)
    parser.add_argument("--max-token-error", type=float, default=0.25)
    parser.add_argument("--max-latency-p95-ms", type=float, default=3500)
    parser.add_argument("--min-language-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = evaluate(report, [lang.strip() for lang in args.languages.split(",") if lang.strip()],
                      args.min_per_language, args.max_cer, args.max_token_error,
                      args.max_latency_p95_ms, args.min_language_accuracy)
    print(json.dumps(result, indent=2))
    return 0 if all(x["pass"] for x in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Controlled failure triage for already-frozen Fixed150 source recordings.

Run on a *small* language-targeted subset with the same Whisper large-v3 model.
Compare the actual app pipeline against raw Whisper and a forced-language
diagnostic. Preserve true local transcripts in E: only; no network upload.

Differences isolate candidate language routing and downstream speech filtering,
not definitive acoustic ground truth. Never use reference-language forcing in
a production benchmark or include forced-language scores in the auto-LID gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from tools.fixed150_prepare import verify_lock, atomic_json
from tools.phase07_benchmark import load_wav


def choose_cases(lock: dict, languages: set[str], max_per_language: int) -> list[dict]:
    cases = [c for c in lock["cases"]
             if c["kind"] == "single" and c["language"] in languages]
    seen: dict[str, int] = {}
    chosen = []
    for case in cases:
        if seen.get(case["language"], 0) >= max_per_language:
            continue
        seen[case["language"]] = seen.get(case["language"], 0) + 1
        chosen.append(case)
    if not chosen:
        raise ValueError("No single-speaker recordings found for requested languages")
    missing = languages - set(seen)
    if missing:
        raise ValueError(f"Languages not in the frozen corpus: {sorted(missing)}")
    return chosen


def score(reference: str, hypothesis: str, language: str) -> dict:
    value = score_pair(reference, hypothesis, language=language)
    return {
        "cer": round(value["cer"], 6),
        "character_edits": value["character_edits"],
        "reference_characters": value["reference_characters"],
        "predicted_characters": len(hypothesis),
        "text": hypothesis,
    }


def direct_transcribe(model, samples, *, language: str | None, use_vad: bool) -> dict:
    options = {
        "language": language,
        "multilingual": language is None,
        "beam_size": 5,
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": use_vad,
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "hallucination_silence_threshold": 1.2,
    }
    if use_vad:
        options["vad_parameters"] = {
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 650,
            "speech_pad_ms": 180,
        }
    started = time.perf_counter()
    segments, info = model.transcribe(samples, **options)
    parts = list(segments)  # Faster-Whisper decodes lazily.
    return {
        "text": " ".join(seg.text.strip() for seg in parts if seg.text.strip()),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments": len(parts),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "diagnostics": [
            {
                "start": seg.start,
                "end": seg.end,
                "no_speech_probability": getattr(seg, "no_speech_prob", None),
                "average_log_probability": getattr(seg, "avg_logprob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
            }
            for seg in parts
        ],
    }


def likely_failure(pipeline: float, raw_auto: float, raw_fixed: float,
                   auto_code: str | None, expected: str) -> list[str]:
    flags = []
    if pipeline - raw_auto >= 0.12:
        flags.append("PIPELINE_LOSS_OR_SEGMENTATION")
    if raw_auto - raw_fixed >= 0.12:
        flags.append("AUTO_LANGUAGE_OR_DECODING")
    if auto_code and auto_code != expected:
        flags.append("WRONG_AUTO_LANGUAGE_LABEL")
    if raw_fixed >= .25:
        flags.append("FORCED_LANGUAGE_STILL_POOR_CHECK_AUDIO_REFERENCE")
    return flags or ["NO_CLEAR_ABLATION_WINNER"]


def analyze(root: Path, language_codes: set[str], max_per_language: int,
            compare_no_vad: bool, model_id: str, output: Path,
            compare_window_modes: bool = False) -> dict:
    from speech.whisper import WhisperEngine

    lock = verify_lock(root)
    selected = choose_cases(lock, language_codes, max_per_language)
    model_engine = WhisperEngine(model_id, "cuda", "float16")
    began = time.perf_counter()
    model = model_engine._get_model()
    preload_ms = round((time.perf_counter() - began) * 1000, 2)
    rows = []
    for i, case in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {case['id']} {case['language']} "
              f"{case['duration_s']:.1f}s", flush=True)
        samples = load_wav(root / case["audio"])
        start = time.perf_counter()
        # The primary pipeline is pinned to the production default whenever
        # this experiment compares the two routing modes. No setting persists.
        with patch.dict(os.environ, {"MEETING_WHISPER_WINDOW_LID": "1"} if compare_window_modes else {}):
            predicted = model_engine.recognize(samples, language="auto")
        pipeline_ms = round((time.perf_counter() - start) * 1000, 2)
        pipeline_text = " ".join(p.text for p in predicted.utterances)
        global_prediction = None
        global_ms = None
        if compare_window_modes:
            started_global = time.perf_counter()
            with patch.dict(os.environ, {"MEETING_WHISPER_WINDOW_LID": "0"}):
                global_prediction = model_engine.recognize(samples, language="auto")
            global_ms = round((time.perf_counter() - started_global) * 1000, 2)
        raw = direct_transcribe(model, samples, language=None, use_vad=True)
        forced = direct_transcribe(model, samples, language=case["language"], use_vad=True)
        no_vad = (direct_transcribe(model, samples, language=None, use_vad=False)
                  if compare_no_vad else None)
        base = score(case["reference"], pipeline_text, case["language"])
        simple = score(case["reference"], raw["text"], case["language"])
        fixed = score(case["reference"], forced["text"], case["language"])
        row = {
            "id": case["id"],
            "source_file": case["audio"],
            "source_sha256": case["sha256"],
            "language": case["language"], "locale": case["locale"],
            "reference": case["reference"],
            "pipeline_auto": {
                **base, "language": predicted.language,
                "segments": len(predicted.utterances),
                "latency_ms": pipeline_ms,
            },
            "raw_auto": {
                **simple,
                "language": raw["language"],
                "language_probability": raw["language_probability"],
                "latency_ms": raw["latency_ms"],
                "segments": raw["segments"],
                "diagnostics": raw["diagnostics"],
            },
            "raw_forced_reference_language": {
                **fixed, "language": forced["language"],
                "latency_ms": forced["latency_ms"],
                "segments": forced["segments"],
                "diagnostics": forced["diagnostics"],
            },
            "suspected_components": likely_failure(
                base["cer"], simple["cer"], fixed["cer"], raw["language"],
                case["language"],
            ),
        }
        if global_prediction is not None:
            global_text = " ".join(p.text for p in global_prediction.utterances)
            row["pipeline_global_language"] = {
                **score(case["reference"], global_text, case["language"]),
                "language": global_prediction.language,
                "segments": len(global_prediction.utterances),
                "latency_ms": global_ms,
            }
            row["window_vs_global_cer_delta_pp"] = round(
                100 * (base["cer"] - row["pipeline_global_language"]["cer"]), 2
            )
        if no_vad is not None:
            row["raw_auto_without_vad"] = {
                **score(case["reference"], no_vad["text"], case["language"]),
                "language": no_vad["language"],
                "language_probability": no_vad["language_probability"],
                "latency_ms": no_vad["latency_ms"],
            }
        rows.append(row)
        global_string = (
            f" global_pipeline={row['pipeline_global_language']['cer']:.1%}"
            if global_prediction is not None else ""
        )
        print(
            f"  CER pipeline={base['cer']:.1%}"
            f"{global_string} "
            f"raw_auto={simple['cer']:.1%} "
            f"forced={fixed['cer']:.1%} "
            f"detected={raw['language']} "
            f"flags={','.join(row['suspected_components'])}",
            flush=True,
        )
    report = {
        "schema": "meeting-fixed150-ablation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "device": "cuda",
        "compute_type": "float16",
        "model_load_ms": preload_ms,
        "compared_window_modes": compare_window_modes,
        "cases": rows,
        "methodology": [
            "Use frozen WAVs and their licensed reference transcripts.",
            "Pipeline auto calls WhisperEngine.recognize including VAD and quality guard.",
            "Raw auto calls Whisper with VAD, but without the app's custom verification.",
            "Optional global-language mode is the same WhisperEngine with window LID disabled.",
            "The global-language comparison is only valid for single-language utterances.",
            "Do not globally disable per-window language detection for code-switched meetings without real tests.",
            "Forced reference language is a diagnostic oracle and NOT a production result.",
            "Optional no-VAD result is a controlled ablation, not a recommended default.",
            "CER differences are clues; listen to audio and examine actual words before fixing models.",
        ],
    }
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--languages", nargs="+", default=["ko"],
                        help="Existing reference codes, e.g. ko pt bn th")
    parser.add_argument("--max-per-language", type=int, default=4)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--no-vad-control", action="store_true")
    parser.add_argument("--compare-window-modes", action="store_true",
                        help="Compare the same production Whisper engine with per-window LID on and off")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_per_language <= 50:
        parser.error("--max-per-language must be between 1 and 50")
    dest = args.output or args.root / "local_failure_ablation.json"
    result = analyze(
        args.root, set(args.languages), args.max_per_language,
        args.no_vad_control, args.model, dest,
        compare_window_modes=args.compare_window_modes,
    )
    print("\nRecorded model comparisons:", len(result["cases"]))
    print("Local diagnostic file (includes source and model transcripts):", dest)
    print("Do not publish the local file without checking source licensing and privacy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

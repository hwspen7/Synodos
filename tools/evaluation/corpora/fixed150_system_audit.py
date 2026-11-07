"""Audit systematic ASR pipeline changes across every frozen single-speaker case.

Compares equal model/audio input through three paths:
  current production window-LID + secondary VAD + quality guard
  same production WhisperEngine with window-LID off
  direct faster-whisper, same primary ASR options but no app verification

Results are per recording and checkpointed after each case. No transcripts are
stored, no source labels are modified, and suspected reference defects remain
visible. The controls isolate potential routing/filtering problems, not their
exact cause. A separate overlap benchmark is required for code switching.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from runtime_quality.quality import score_pair
from tools.evaluation.corpora.fixed150_ablation import direct_transcribe
from tools.evaluation.corpora.fixed150_prepare import atomic_json, verify_lock
from tools.evaluation.corpora.fixed150_reference_audit import FOCUS_15, source_flags
from tools.validation.session_quality_benchmark import load_wav

VERSION = "meeting-fixed150-system-ablation-v1"
RELEVANT_SOURCES = (
    "speech/asr/whisper.py", "speech/audio/audio_windows.py",
    "speech/asr/multilingual_windows.py", "speech/captions/quality_guard.py",
    "speech/captions/utterance_units.py", "speech/captions/text_format.py",
    "tools/evaluation/corpora/fixed150_ablation.py",
)


def fingerprint(root: Path, lock: dict, model: str) -> str:
    hasher = sha256()
    hasher.update(VERSION.encode())
    hasher.update(model.encode())
    hasher.update(json.dumps(
        [(item["id"], item["sha256"]) for item in lock["cases"]],
        sort_keys=True, separators=(",", ":")
    ).encode())
    repo = Path(__file__).resolve().parents[3]
    for relative in RELEVANT_SOURCES:
        hasher.update(relative.encode())
        hasher.update((repo / relative).read_bytes())
    return hasher.hexdigest()


def metric(reference: str, text: str, lang: str, label: str, ms: float,
           segments: int) -> dict:
    score = score_pair(reference, text, language=lang)
    return {
        "cer": score["cer"], "character_edits": score["character_edits"],
        "reference_characters": score["reference_characters"],
        "hypothesis_characters": len(text),
        "language": label, "latency_ms": round(ms, 2),
        "segments": segments,
    }


def _speech_geometry(samples) -> dict:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from speech.audio.audio_windows import speech_windows

    vad = get_speech_timestamps(samples, VadOptions(
        min_speech_duration_ms=200, min_silence_duration_ms=650,
        speech_pad_ms=180,
    ), sampling_rate=16000)
    windows = speech_windows(vad, len(samples))
    boundaries = [end for (_, end), _ in zip(windows, windows[1:])]
    # A seam is internal to a VAD interval when it is surrounded by 150 ms
    # of uninterrupted acoustically-detected speech on both sides.
    internal = [b for b in boundaries if any(
        item["start"] + 2400 <= b <= item["end"] - 2400 for item in vad
    )]
    return {
        "vad_intervals": len(vad),
        "vad_speech_seconds": round(sum(
            max(0, x["end"] - x["start"]) for x in vad
        ) / 16000, 3),
        "asr_windows": len(windows),
        "window_boundary_inside_detected_speech": len(internal),
        "boundary_seconds": [round(b / 16000, 3) for b in internal],
    }


def _recognize(engine, samples, reference: str, language: str,
               window_mode: str) -> dict:
    started = time.perf_counter()
    with patch.dict(os.environ, {"MEETING_WHISPER_WINDOW_LID": window_mode,
                                 "MEETING_WHISPER_DUAL_VAD": "0"}):
        result = engine.recognize(samples, language="auto")
    elapsed = (time.perf_counter() - started) * 1000
    return metric(reference, " ".join(x.text for x in result.utterances),
                  language, result.language, elapsed, len(result.utterances))


def test_one(engine, model, root: Path, case: dict) -> dict:
    samples = load_wav(root / case["audio"])
    lang, ref = case["language"], case["reference"]
    window = _recognize(engine, samples, ref, lang, "1")
    global_mode = _recognize(engine, samples, ref, lang, "0")
    raw = direct_transcribe(model, samples, language=None, use_vad=True)
    direct = metric(ref, raw["text"], lang, raw["language"],
                    raw["latency_ms"], raw["segments"])
    diffs = {
        "window_minus_global_pp": round(
            100 * (window["cer"] - global_mode["cer"]), 3),
        "window_minus_raw_pp": round(
            100 * (window["cer"] - direct["cer"]), 3),
        "global_minus_raw_pp": round(
            100 * (global_mode["cer"] - direct["cer"]), 3),
    }
    return {
        "id": case["id"], "sha256": case["sha256"],
        "language": lang, "locale": case.get("locale"),
        "scenario": case["scenario"], "duration_s": case["duration_s"],
        "reference_review_flags": source_flags(case["duration_s"], ref),
        "acoustic_geometry": _speech_geometry(samples),
        "production_windowed": window,
        "production_whole_clip": global_mode,
        "direct_whisper": direct,
        "delta": diffs,
    }


def sum_mode(rows: list[dict], name: str) -> float | None:
    den = sum(x[name]["reference_characters"] for x in rows)
    return (sum(x[name]["character_edits"] for x in rows) / den
            if den else None)


def summarize(rows: list[dict], total_requested: int) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    modes = ("production_windowed", "production_whole_clip", "direct_whisper")
    all_group = {}
    for language in sorted(groups):
        cases = groups[language]
        all_group[language] = {
            "samples": len(cases),
            "cer": {mode: sum_mode(cases, mode) for mode in modes},
            "window_worse_than_whole_5pp": sum(
                x["delta"]["window_minus_global_pp"] >= 5 for x in cases),
            "window_worse_than_raw_5pp": sum(
                x["delta"]["window_minus_raw_pp"] >= 5 for x in cases),
            "window_better_than_raw_5pp": sum(
                x["delta"]["window_minus_raw_pp"] <= -5 for x in cases),
            "source_reference_review": sum(
                bool(x["reference_review_flags"]) for x in cases),
            "boundary_inside_vad_speech": sum(
                x["acoustic_geometry"]["window_boundary_inside_detected_speech"]
                for x in cases),
        }
    return {
        "complete": len(rows) == total_requested,
        "completed": len(rows), "requested": total_requested,
        "overall_cer": {mode: sum_mode(rows, mode) for mode in modes},
        "cases_window_worse_than_raw_5pp": sum(
            x["delta"]["window_minus_raw_pp"] >= 5 for x in rows),
        "cases_window_better_than_raw_5pp": sum(
            x["delta"]["window_minus_raw_pp"] <= -5 for x in rows),
        "cases_window_worse_than_whole_5pp": sum(
            x["delta"]["window_minus_global_pp"] >= 5 for x in rows),
        "cases_with_speech_boundary_cuts": sum(
            x["acoustic_geometry"]["window_boundary_inside_detected_speech"] > 0
            for x in rows),
        "cases_requiring_reference_review": sum(
            bool(x["reference_review_flags"]) for x in rows),
        "per_language": all_group,
    }


def run(root: Path, scope: str, output: Path, model_id: str, limit: int | None) -> dict:
    from speech.asr.whisper import WhisperEngine

    lock = verify_lock(root)
    singles = [x for x in lock["cases"] if x["kind"] == "single"]
    if len(singles) != 100:
        raise ValueError("Exactly 100 original single-speaker recordings required")
    focus = set(FOCUS_15)
    selected = [x for x in singles if scope == "all" or x["language"] in focus]
    expected = 100 if scope == "all" else 68
    if len(selected) != expected:
        raise ValueError(f"Unexpected source allocation: {len(selected)} != {expected}")

    fpr = fingerprint(root, lock, model_id)
    state = {
        "schema": VERSION,
        "scope": scope, "fingerprint": fpr,
        "model": model_id,
        "codes": sorted({x["language"] for x in selected}),
        "cases": {},
    }
    if output.exists():
        state = json.loads(output.read_text(encoding="utf-8"))
        if (state.get("schema") != VERSION or state.get("scope") != scope or
            state.get("fingerprint") != fpr):
            raise ValueError(
                "Existing result belongs to a different code, model or scope. "
                "Use a different --output; never mix incompatible measurements."
            )
    for case in selected:
        existing = state["cases"].get(case["id"])
        if existing and existing["sha256"] != case["sha256"]:
            raise ValueError(f"Checkpoint source mismatch: {case['id']}")

    pending = [c for c in selected if c["id"] not in state["cases"]]
    if limit is not None:
        pending = pending[:limit]
    if pending:
        engine = WhisperEngine(model_id, "cuda", "float16")
        model = engine._get_model()
        for case in pending:
            print(f"[{len(state['cases'])+1}/{len(selected)}] "
                  f"{case['id']} {case['language']} {case['duration_s']:.1f}s",
                  flush=True)
            row = test_one(engine, model, root, case)
            state["cases"][case["id"]] = row
            atomic_json(output, state)
            print(
                "  window {0:.1%} | global {1:.1%} | raw {2:.1%} | "
                "speech-cut boundaries {3}".format(
                    row["production_windowed"]["cer"],
                    row["production_whole_clip"]["cer"],
                    row["direct_whisper"]["cer"],
                    row["acoustic_geometry"]["window_boundary_inside_detected_speech"],
                ), flush=True,
            )
    ordered = [state["cases"][x["id"]] for x in selected
               if x["id"] in state["cases"]]
    summary = summarize(ordered, len(selected))
    summary_path = output.with_name(output.stem + "_summary.json")
    atomic_json(summary_path, {
        "schema": VERSION + "-summary", "scope": scope,
        "fingerprint": fpr, **summary,
        "note": ("Scores are paired model ablations, not a product release gate. "
                 "Review suspicious source labels manually. "
                 "Global-language mode is a monolingual control, not a "
                 "replacement for mixed-language meeting decoding."),
    })
    print("\nStatus:", "COMPLETE" if summary["complete"] else "PARTIAL",
          f'{summary["completed"]}/{summary["requested"]}')
    print("Window worse than RAW by >=5 percentage points:",
          summary["cases_window_worse_than_raw_5pp"])
    print("Window worse than GLOBAL by >=5 percentage points:",
          summary["cases_window_worse_than_whole_5pp"])
    print("Recordings with cut boundaries inside detected speech:",
          summary["cases_with_speech_boundary_cuts"])
    print("\nLanguage    n     window     global        raw   5pp harmed")
    for lang, data in summary["per_language"].items():
        c = data["cer"]
        print(f'{lang:8} {data["samples"]:3} '
              f'{c["production_windowed"]:10.2%} '
              f'{c["production_whole_clip"]:10.2%} '
              f'{c["direct_whisper"]:10.2%} '
              f'{data["window_worse_than_raw_5pp"]:5}')
    print("Checkpoint:", output)
    print("Summary:", summary_path)
    if not summary["complete"]:
        print("Results are incomplete; rerun the same command to resume.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--scope", choices=("focus", "all"), default="focus")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = args.output or (
        args.root / f"system_ablation_{args.scope}.json")
    result = run(args.root, args.scope, output, args.model, args.limit)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

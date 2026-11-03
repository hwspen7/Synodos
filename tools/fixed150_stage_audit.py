"""Isolate VAD and post-ASR effects on already-measured regression cases.

This is a controlled, local diagnostic only. It does not change the production
pipeline, overwrite frozen references, or claim a model improvement. By default
only numeric statistics are retained (not transcripts). Each successful clip
is checkpointed and bound to source hashes and the relevant implementation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from tools.fixed150_prepare import atomic_json, verify_lock
from tools.phase07_benchmark import load_wav
from speech.audio_windows import local_speech_regions, speech_windows
from speech.quality_guard import verify_utterances
from speech.utterance_units import split_utterances


SCHEMA = "meeting-fixed150-stage-audit-v2"
SOURCES = (
    "speech/whisper.py", "speech/multilingual_windows.py",
    "speech/audio_windows.py", "speech/quality_guard.py",
    "speech/utterance_units.py", "tools/fixed150_stage_audit.py",
)


def select_impacted(lock: dict, system_report: dict, *, threshold_pp: float = 5.0) -> list[dict]:
    if system_report.get("schema") != "meeting-fixed150-system-ablation-v1":
        raise ValueError("Unexpected system ablation schema")
    rows = system_report.get("cases", {})
    if not isinstance(rows, dict) or len(rows) != 100:
        raise ValueError("Require completed 100-original system ablation")
    singles = {x["id"]: x for x in lock["cases"] if x["kind"] == "single"}
    if len(singles) != 100:
        raise ValueError("Incorrect source allocation")
    impacted = []
    for cid, row in rows.items():
        if cid not in singles or row.get("sha256") != singles[cid]["sha256"]:
            raise ValueError(f"Unverified system-ablation source {cid}")
        window = row["production_windowed"]["cer"]
        direct = row["direct_whisper"]["cer"]
        if (window - direct) * 100 >= threshold_pp:
            impacted.append((100 * (window - direct), singles[cid]))
    # Triage the most severe measured regressions before less costly ones.
    return [case for _, case in sorted(
        impacted, key=lambda item: (-item[0], item[1]["id"])
    )]


def fingerprint(lock: dict, model: str, scope: str, threshold_pp: float) -> str:
    repo = Path(__file__).resolve().parents[1]
    hasher = sha256()
    for value in (SCHEMA, model, scope, str(threshold_pp)):
        hasher.update(value.encode())
    hasher.update(json.dumps(
        [(c["id"], c["sha256"]) for c in lock["cases"]],
        sort_keys=True, separators=(",", ":")
    ).encode())
    for path in SOURCES:
        hasher.update(path.encode())
        hasher.update((repo / path).read_bytes())
    return hasher.hexdigest()


def measure(reference: str, hypothesis: str, language: str) -> dict:
    result = score_pair(reference, hypothesis, language=language)
    return {
        "cer": result["cer"], "character_edits": result["character_edits"],
        "reference_characters": result["reference_characters"],
        "predicted_characters": len(hypothesis),
    }


def run_one(model, case: dict, root: Path) -> dict:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    samples = load_wav(root / case["audio"])
    options = VadOptions(
        min_speech_duration_ms=200, min_silence_duration_ms=650,
        speech_pad_ms=180,
    )
    regions = get_speech_timestamps(samples, options, sampling_rate=16000)
    windows = speech_windows(regions, len(samples), sample_rate=16000)
    if len(windows) > 100:
        raise RuntimeError("Unexpectedly many ASR windows")
    defaults = {
        "language": None, "multilingual": True, "beam_size": 5,
        "word_timestamps": True,
        "vad_parameters": {
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 650,
            "speech_pad_ms": 180,
        },
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "hallucination_silence_threshold": 1.2,
        "condition_on_previous_text": False,
        "temperature": 0,
    }
    outputs = {"window_raw_second_vad": [], "window_verified_second_vad": [],
               "window_preserved_second_vad": [],
               "window_raw_no_second_vad": [], "window_verified_no_second_vad": [],
               "window_preserved_no_second_vad": []}
    window_details = []
    start_test = time.perf_counter()
    for start, end in windows:
        local = local_speech_regions(regions, start, end)
        if not local:
            continue
        record = {"start_ms": round(start / 16), "end_ms": round(end / 16)}
        for second_vad in (True, False):
            params = dict(defaults, vad_filter=second_vad)
            # Consume model generator before working on later windows.
            pieces, info = model.transcribe(samples[start:end], **params)
            parts = list(pieces)
            detected = str(getattr(info, "language", "und") or "und").lower()
            probability = getattr(info, "language_probability", None)
            accepted = verify_utterances(
                parts, local, round((end - start) / 16), detected
            )
            preserved = verify_utterances(
                parts, local, round((end - start) / 16), detected,
                preserve_disputed_text=True,
            )
            accepted = split_utterances(tuple(accepted))
            preserved = split_utterances(tuple(preserved))
            suffix = "second_vad" if second_vad else "no_second_vad"
            raw_text = " ".join(str(getattr(p, "text", "") or "").strip()
                                for p in parts if str(getattr(p, "text", "") or "").strip())
            accepted_text = " ".join(p.text for p in accepted)
            preserved_text = " ".join(p.text for p in preserved)
            outputs["window_raw_" + suffix].append(raw_text)
            outputs["window_verified_" + suffix].append(accepted_text)
            outputs["window_preserved_" + suffix].append(preserved_text)
            record[suffix] = {
                "language": detected,
                "language_probability": probability,
                "raw_model_segments": len(parts),
                "accepted_utterances": len(accepted),
                "raw_characters": len(raw_text),
                "accepted_characters": len(accepted_text),
                "preserved_characters": len(preserved_text),
                "preserved_without_word_timing": sum(
                    item.timestamp_quality == "segment" for item in preserved
                ),
                "raw_words_with_timestamps": sum(len(getattr(p, "words", None) or [])
                                                 for p in parts),
                "accepted_words": sum(len(p.words) for p in accepted),
            }
        window_details.append(record)
    report = {
        "id": case["id"],
        "language": case["language"],
        "sha256": case["sha256"],
        "duration_s": case["duration_s"],
        "windows": len(windows),
        "window_details": window_details,
        "seconds_elapsed": round(time.perf_counter() - start_test, 2),
    }
    for key, parts in outputs.items():
        transcript = " ".join(x for x in parts if x)
        report[key] = measure(case["reference"], transcript, case["language"])
    return report


def run(root: Path, source_path: Path, output: Path,
        scope: str, model_id: str, threshold_pp: float, limit: int | None):
    from speech.whisper import WhisperEngine

    lock = verify_lock(root)
    base = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = select_impacted(lock, base, threshold_pp=threshold_pp)
    if scope == "focus":
        from tools.fixed150_reference_audit import FOCUS_15
        candidates = [c for c in candidates if c["language"] in set(FOCUS_15)]
    if not candidates:
        raise ValueError("No impacted cases in selected scope")
    key = fingerprint(lock, model_id, scope, threshold_pp)
    state = {"schema": SCHEMA, "fingerprint": key, "scope": scope, "model": model_id,
             "threshold_pp": threshold_pp, "cases": {}}
    if output.exists():
        state = json.loads(output.read_text(encoding="utf-8"))
        if state.get("fingerprint") != key or state.get("schema") != SCHEMA:
            raise ValueError("Existing stage audit belongs to different inputs or code")
    for case in candidates:
        old = state["cases"].get(case["id"])
        if old and old["sha256"] != case["sha256"]:
            raise ValueError(f"Source mismatch: {case['id']}")
    pending = [c for c in candidates if c["id"] not in state["cases"]]
    if limit is not None:
        pending = pending[:limit]
    if pending:
        engine = WhisperEngine(model_id, "cuda", "float16")
        model = engine._get_model()
        for case in pending:
            print(f"[{len(state['cases']) + 1}/{len(candidates)}] "
                  f"{case['id']} {case['language']} {case['duration_s']:.1f}s",
                  flush=True)
            row = run_one(model, case, root)
            state["cases"][case["id"]] = row
            atomic_json(output, state)
            for name in ("window_raw_second_vad", "window_verified_second_vad",
                         "window_preserved_second_vad",
                         "window_raw_no_second_vad", "window_verified_no_second_vad",
                         "window_preserved_no_second_vad"):
                print(f"  {name:30} CER={row[name]['cer']:.2%}", flush=True)
    done = len(state["cases"])
    print(f"Stage-audit status: {done}/{len(candidates)}")
    print("Checkpoint:", output)
    print("Original annotations and ASR model outputs were not modified.")
    return done == len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark/fixed150/system_ablation_all.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/fixed150/stage_guard_ablation.json"))
    parser.add_argument("--scope", choices=("focus", "all"), default="all")
    parser.add_argument("--threshold-pp", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()
    if args.threshold_pp <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("Threshold and limit must be positive")
    ok = run(args.root, args.source, args.output, args.scope, args.model,
             args.threshold_pp, args.limit)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

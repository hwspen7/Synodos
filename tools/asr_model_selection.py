"""Paired multilingual ASR selection on the same immutable recorded WAVs.

FLEURS provides external published transcripts, not independently listened
meeting gold. Outputs are private and never change the production ASR engine.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata
import wave

from tools.freeze_fleurs_asr_corpus import (
    LANGUAGES, SCHEMA as CORPUS_SCHEMA, ROOT, configure_cache,
    require_e_drive,
)

PREDICTIONS_SCHEMA = "meeting-asr-selection-predictions-v1"
SCORE_SCHEMA = "meeting-asr-selection-score-v1"
COMPARISON_SCHEMA = "meeting-asr-selection-comparison-v1"
MODELS = {
    "whisper-large-v3": ("whisper", "large-v3"),
    "qwen3-asr-1.7b": ("qwen3", "Qwen/Qwen3-ASR-1.7B"),
    "qwen3-asr-0.6b": ("qwen3", "Qwen/Qwen3-ASR-0.6B"),
    "hybrid-whisper-qwen17": (
        "hybrid", "Whisper-large-v3+Qwen3-ASR-1.7B/fleurs-validation-200-v1"
    ),
}
CJK_CER = {"zh", "ja"}
HEX64 = re.compile(r"[0-9a-f]{64}")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_corpus(path: Path) -> tuple[dict, str]:
    path = require_e_drive(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema") != CORPUS_SCHEMA or
            doc.get("dataset") != "google/fleurs" or
            doc.get("reference_is_model_independent") is not True or
            doc.get("model_scores_used_to_choose_cases") is not False or
            doc.get("production_release_approved") is not False or
            doc.get("data_split") not in ("validation", "test") or
            set(doc.get("languages", [])) != set(LANGUAGES) or
            type(doc.get("per_language")) is not int):
        raise ValueError("Untrusted multilingual corpus metadata")
    needed = doc["per_language"]
    if not 10 <= needed <= 400:
        raise ValueError("Invalid per-language count")
    if doc.get("benchmark_role") != (
        "development" if doc["data_split"] == "validation" else "sealed_test"):
        raise ValueError("Invalid source split")
    if doc["data_split"] == "test" and needed < 50:
        raise ValueError("Sealed test requires at least 50 clips per language")
    cases = doc.get("cases")
    if not isinstance(cases, list) or len(cases) != needed * len(LANGUAGES):
        raise ValueError("Missing or additional corpus cases")
    by_lang, seen = defaultdict(int), set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Malformed corpus case")
        language, case_id = item.get("language"), item.get("id")
        if (language not in LANGUAGES or
                not isinstance(case_id, str) or not case_id or case_id in seen):
            raise ValueError("Invalid or duplicate case ID/language")
        seen.add(case_id)
        by_lang[language] += 1
        if (item.get("split") != doc["benchmark_role"] or
                item.get("reference_kind") != "published_transcript" or
                item.get("independently_audio_reviewed") is not False or
                item.get("selected_independently_of_asr") is not True or
                not isinstance(item.get("reference_text"), str) or
                not item["reference_text"].strip() or
                not isinstance(item.get("reference_source"), str) or
                not isinstance(item.get("audio_sha256"), str) or
                not HEX64.fullmatch(item["audio_sha256"])):
            raise ValueError("Missing independent official reference provenance")
        wav = require_e_drive(Path(item["audio_path"]))
        if not wav.is_relative_to(path.parent) or not wav.is_file():
            raise ValueError("All frozen WAV files must be inside the corpus directory")
        if file_hash(wav) != item["audio_sha256"]:
            raise ValueError("Audio checksum mismatch for " + case_id)
        with wave.open(str(wav), "rb") as input:
            if (input.getframerate(), input.getnchannels(), input.getsampwidth(),
                    input.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("Incompatible WAV encoding for " + case_id)
            actual_duration = input.getnframes() / 16000
            if not .8 <= actual_duration <= 30 or abs(
                    actual_duration - item["audio_duration_seconds"]) > .002:
                raise ValueError("Invalid stored source duration")
    if set(by_lang) != set(LANGUAGES) or any(by_lang[x] != needed for x in LANGUAGES):
        raise ValueError("Uneven language cohort")
    return doc, file_hash(path)


def load_audio(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        audio = wav.readframes(wav.getnframes())
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0


def engine_for(model: str):
    backend, identifier = MODELS[model]
    if backend == "whisper":
        from speech.whisper import WhisperEngine
        return WhisperEngine(identifier, device="cuda", compute_type="float16")
    if backend == "hybrid":
        from speech.hybrid import HybridEngine
        from speech.qwen_bridge import QwenWorkerClient
        from speech.whisper import WhisperEngine
        python = os.environ.get("MEETING_HYBRID_QWEN_PYTHON", "")
        if not python:
            raise ValueError("Set MEETING_HYBRID_QWEN_PYTHON for a paired hybrid trial")
        return HybridEngine(
            WhisperEngine("large-v3", device="cuda", compute_type="float16"),
            QwenWorkerClient(python, "Qwen/Qwen3-ASR-1.7B"),
        )
    from speech.qwen import QwenEngine
    return QwenEngine(identifier, device="cuda:0", aligner=False)


def infer_one(engine, audio, language):
    answer = engine.recognize(audio, language=language)
    return " ".join(u.text.strip() for u in answer.utterances if u.text.strip()), (
        answer.language or "und")


def record_predictions(manifest_path: Path, model: str, output: Path,
                       language_mode: str) -> dict:
    if model not in MODELS or language_mode not in ("auto", "known"):
        raise ValueError("Unsupported ASR model or comparison mode")
    manifest, sha = load_corpus(manifest_path)
    output = require_e_drive(output)
    if output.exists():
        raise FileExistsError("ASR prediction output is immutable")
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("ASR prediction file must stay in frozen E-drive benchmark root")
    configure_cache(ROOT)
    engine = engine_for(model)
    records = []
    for index, case in enumerate(manifest["cases"]):
        samples = load_audio(Path(case["audio_path"]))
        chosen_language = case["language"] if language_mode == "known" else "auto"
        started = time.perf_counter()
        error, text, detected = "", "", "und"
        try:
            text, detected = infer_one(engine, samples, chosen_language)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)[:180]
        seconds = round(time.perf_counter() - started, 4)
        records.append({
            "id": case["id"], "audio_sha256": case["audio_sha256"],
            "text": text, "detected_language": detected,
            "latency_seconds": seconds,
            "first_call_includes_model_load": index == 0,
            "error": error,
        })
        print("ASR", model, index + 1, "of", len(manifest["cases"]),
              "FAILED" if error else "OK", flush=True)
    result = {
        "schema": PREDICTIONS_SCHEMA, "model": model,
        "backend": MODELS[model][0], "model_id": MODELS[model][1],
        "language_mode": language_mode,
        "manifest_sha256": sha, "manifest_path": str(manifest_path.resolve()),
        "benchmark_role": manifest["benchmark_role"],
        "observations": records,
        "any_case_failed": any(record["error"] for record in records),
        "reference_transcript_used_during_inference": False,
        "production_asr_modified": False, "may_auto_promote": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    return result


def normalize(text):
    raw = unicodedata.normalize("NFKC", text).casefold()
    # Preserve Unicode combining marks, e.g. Indic vowel signs, which
    # ordinary regex word matching would incorrectly split or discard.
    cleaned = []
    for char in raw:
        kind = unicodedata.category(char)[0]
        if kind in "LNM" or char.isspace():
            cleaned.append(char)
        elif char in ("'", "’"):
            continue
        else:
            cleaned.append(" ")
    words = "".join(cleaned).split()
    return words, list("".join(words))


def distance(a, b):
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, token in enumerate(b, 1):
        cur = [i]
        for j, other in enumerate(a, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j-1] + (token != other)))
        prev = cur
    return prev[-1]


def p95(data):
    if not data:
        return None
    sorted_data = sorted(data)
    return round(sorted_data[min(len(data) - 1, math.ceil(.95 * len(data)) - 1)], 4)


def score_predictions(manifest_path, prediction_path):
    corpus, manifest_sha = load_corpus(manifest_path)
    prediction = json.loads(require_e_drive(prediction_path).read_text(encoding="utf-8"))
    if (prediction.get("schema") != PREDICTIONS_SCHEMA or
            prediction.get("model") not in MODELS or
            prediction.get("model_id") != MODELS[prediction["model"]][1] or
            prediction.get("manifest_sha256") != manifest_sha or
            prediction.get("benchmark_role") != corpus["benchmark_role"] or
            prediction.get("language_mode") not in ("auto", "known") or
            prediction.get("reference_transcript_used_during_inference") is not False or
            prediction.get("production_asr_modified") is not False):
        raise ValueError("Untrusted ASR prediction provenance")
    actual = prediction.get("observations")
    cases = corpus["cases"]
    if not isinstance(actual, list) or len(actual) != len(cases):
        raise ValueError("ASR output cohort incomplete")
    totals = defaultdict(lambda: {
        "cases": 0, "primary_reference_units": 0, "primary_edits": 0,
        "word_reference_units": 0, "word_edits": 0,
        "character_reference_units": 0, "character_edits": 0,
        "empty_predictions": 0, "failures": 0,
        "latencies": [], "rtfs": [], "language_matches": 0,
    })
    for case, item in zip(cases, actual):
        if (item.get("id") != case["id"] or
                item.get("audio_sha256") != case["audio_sha256"] or
                not isinstance(item.get("text"), str) or
                not isinstance(item.get("error"), str) or
                type(item.get("latency_seconds")) not in (float, int) or
                not math.isfinite(item["latency_seconds"]) or
                item["latency_seconds"] < 0):
            raise ValueError("Unpaired or invalid model result for frozen audio")
        stats = totals[case["language"]]
        stats["cases"] += 1
        ref_words, ref_chars = normalize(case["reference_text"])
        pred_words, pred_chars = normalize(item["text"])
        word_edits = distance(ref_words, pred_words)
        character_edits = distance(ref_chars, pred_chars)
        if not ref_words or not ref_chars:
            raise ValueError("Empty normalized published reference")
        is_cjk = case["language"] in CJK_CER
        stats["primary_reference_units"] += len(ref_chars) if is_cjk else len(ref_words)
        stats["primary_edits"] += character_edits if is_cjk else word_edits
        stats["word_reference_units"] += len(ref_words)
        stats["word_edits"] += word_edits
        stats["character_reference_units"] += len(ref_chars)
        stats["character_edits"] += character_edits
        stats["empty_predictions"] += int(not item["text"].strip())
        stats["failures"] += int(bool(item["error"]))
        stats["language_matches"] += int(item.get("detected_language") == case["language"])
        if not item.get("first_call_includes_model_load") and not item["error"]:
            stats["latencies"].append(item["latency_seconds"])
            stats["rtfs"].append(item["latency_seconds"] / case["audio_duration_seconds"])
    per_language = {}
    for lang in LANGUAGES:
        stats = totals[lang]
        if stats["cases"] != corpus["per_language"] or not stats["primary_reference_units"]:
            raise ValueError("Incomplete language comparison")
        per_language[lang] = {
            "cases": stats["cases"],
            "primary_unit": "CER" if lang in CJK_CER else "WER",
            "primary_error_rate": round(stats["primary_edits"] / stats["primary_reference_units"], 5),
            "wer": round(stats["word_edits"] / stats["word_reference_units"], 5),
            "cer": round(stats["character_edits"] / stats["character_reference_units"], 5),
            "empty_predictions": stats["empty_predictions"],
            "failed_cases": stats["failures"],
            "language_id_accuracy": round(stats["language_matches"] / stats["cases"], 4)
                if prediction["language_mode"] == "auto" else None,
            "mean_realtime_factor_excluding_cold_start": round(statistics.mean(stats["rtfs"]), 5)
                if stats["rtfs"] else None,
            "p95_latency_s_excluding_cold_start": p95(stats["latencies"]),
        }
    rates = [per_language[x]["primary_error_rate"] for x in LANGUAGES]
    total_failures = sum(item["failed_cases"] for item in per_language.values())
    total_empty = sum(item["empty_predictions"] for item in per_language.values())
    return {
        "schema": SCORE_SCHEMA,
        "source_manifest_sha256": manifest_sha,
        "source_predictions_sha256": file_hash(require_e_drive(prediction_path)),
        "model": prediction["model"],
        "language_mode": prediction["language_mode"],
        "benchmark_role": corpus["benchmark_role"],
        "evaluated_cases": len(cases),
        "per_language_case_count": corpus["per_language"],
        "primary_definition": "Equal-weight macro CER for zh/ja and WER for other 8 languages",
        "macro_primary_error_rate": round(statistics.mean(rates), 5),
        "worst_language_error_rate": round(max(rates), 5),
        "empty_prediction_count": total_empty,
        "failed_case_count": total_failures,
        "per_language": per_language,
        "eligible_for_read_speech_ranking": total_failures == 0,
        "live_meeting_gold_evaluated": False,
        "speaker_diarization_evaluated": False,
        "production_release_pass": False,
        "may_auto_promote": False,
    }


def compare_reports(paths):
    if len(paths) < 2:
        raise ValueError("At least two different model score reports required")
    reports = [json.loads(require_e_drive(p).read_text(encoding="utf-8")) for p in paths]
    first = reports[0]
    for row in reports:
        if (row.get("schema") != SCORE_SCHEMA or
                row.get("source_manifest_sha256") != first.get("source_manifest_sha256") or
                row.get("language_mode") != first.get("language_mode") or
                row.get("benchmark_role") != first.get("benchmark_role") or
                row.get("evaluated_cases") != first.get("evaluated_cases") or
                row.get("per_language_case_count") != first.get("per_language_case_count") or
                set(row.get("per_language", {})) != set(LANGUAGES) or
                row.get("live_meeting_gold_evaluated") is not False or
                row.get("production_release_pass") is not False):
            raise ValueError("Incompatible model trial or missing benchmark provenance")
    if len({r.get("model") for r in reports}) != len(reports):
        raise ValueError("Duplicate model trials")
    ranked = sorted(reports, key=lambda r: (
        not r["eligible_for_read_speech_ranking"],
        r["macro_primary_error_rate"],
        r["worst_language_error_rate"],
        r["empty_prediction_count"],
        r["model"],
    ))
    eligible = [row for row in ranked if row["eligible_for_read_speech_ranking"]]
    return {
        "schema": COMPARISON_SCHEMA,
        "source_manifest_sha256": first["source_manifest_sha256"],
        "language_mode": first["language_mode"],
        "benchmark_role": first["benchmark_role"],
        "ranked_models": [{
            "model": row["model"],
            "macro_primary_error_rate": row["macro_primary_error_rate"],
            "worst_language_error_rate": row["worst_language_error_rate"],
            "empty_prediction_count": row["empty_prediction_count"],
            "failed_case_count": row["failed_case_count"],
            "eligible_for_read_speech_ranking": row["eligible_for_read_speech_ranking"],
            "source_predictions_sha256": row["source_predictions_sha256"],
        } for row in ranked],
        "best_on_this_published_read_speech_cohort":
            eligible[0]["model"] if eligible else None,
        "best_production_meeting_model": None,
        "production_release_pass": False,
        "may_auto_promote": False,
        "caution": "Public read speech is not human-reviewed live meeting speech; do not change production defaults from this report.",
    }


def write_new(path: Path, data: dict):
    path = require_e_drive(path)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Write private results under the E-drive ASRSelection root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    validate = sp.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    run = sp.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--model", choices=sorted(MODELS), required=True)
    run.add_argument("--language-mode", choices=["auto", "known"], default="auto")
    run.add_argument("--output", required=True, type=Path)
    score = sp.add_parser("score")
    score.add_argument("--manifest", required=True, type=Path)
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    compare = sp.add_parser("compare")
    compare.add_argument("--scores", nargs="+", type=Path, required=True)
    compare.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "validate":
            corpus, sha = load_corpus(args.manifest)
            print("CORPUS VERIFIED", len(corpus["cases"]), "samples", sha)
            return 0
        if args.command == "run":
            result = record_predictions(args.manifest, args.model, args.output,
                                        args.language_mode)
            print("ASR RESULT", args.output,
                  "failed_cases", sum(bool(x["error"]) for x in result["observations"]))
            return 2 if result["any_case_failed"] else 0
        if args.command == "score":
            result = score_predictions(args.manifest, args.predictions)
            write_new(args.output, result)
            print("ASR SCORE", result["model"], "macro_primary_error",
                  result["macro_primary_error_rate"],
                  "failed_cases", result["failed_case_count"])
            return 2 if result["failed_case_count"] else 0
        result = compare_reports(args.scores)
        write_new(args.output, result)
        print("READ SPEECH COHORT WINNER", result["best_on_this_published_read_speech_cohort"])
        print("PRODUCTION WINNER NOT YET DETERMINED")
        return 0 if result["best_on_this_published_read_speech_cohort"] else 2
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print("ASR MODEL SELECTION BLOCKED", type(exc).__name__,
              str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

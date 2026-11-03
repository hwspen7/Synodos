"""Evaluate Fixed195 singles quantitatively; review real dialogues qualitatively.

Primary: all 130 frozen single-speaker recordings with gold references.
Secondary: only downloaded public authentic stereo dialogues, without gold
transcripts. Multi output is diagnostic transcription for HUMAN inspection,
not CER, DER, cpWER, intelligibility accuracy, nor an automatic 'human'
judgment. No reference text is fabricated or inferred from ASR predictions.

Uses the production WhisperEngine recognize() pipeline without modifying
runtime defaults. Output is checkpointed per case, validates input hashes,
and includes a model/source/code fingerprint to prevent stale result reuse.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from tools.fixed150_prepare import atomic_json, checksum
from tools.phase07_benchmark import load_wav
from tools.fixed195_prepare import LANGUAGES, singles_for

VERSION = "meeting-fixed195-content-review-v1"
CODE_PATHS = (
    "speech/whisper.py",
    "speech/multilingual_windows.py",
    "speech/quality_guard.py",
    "speech/audio_windows.py",
    "speech/utterance_units.py",
    "speech/text_format.py",
    "phase07/quality.py",
    "tools/fixed195_content_review.py",
)
LANGUAGE_ORDER = [language for language, _ in LANGUAGES]
# These are REVIEW FLAGS, not independent quality scores.
REPEAT_PATTERN = re.compile(r"(?iu)(?:\b(\w{2,})\b[\s,，。.!?]*)\1(?:\b|$)")
MAX_RECENT_UTTERANCES = 160


def load_manifest(path: Path, expected: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != expected or len({x.get("id") for x in rows}) != expected:
        raise ValueError(f"Manifest must hold exactly {expected} distinct cases")
    return rows


def validate_singles(root: Path) -> list[dict]:
    rows = load_manifest(root / "manifest_single.jsonl", 130)
    counts = defaultdict(int)
    for row in rows:
        if row.get("kind") != "single" or row.get("language") not in LANGUAGE_ORDER:
            raise ValueError(f"Invalid single-speaker case: {row.get('id')}")
        if not isinstance(row.get("reference"), str) or not row["reference"].strip():
            raise ValueError(f"Missing human reference: {row.get('id')}")
        if not row.get("id", "").startswith(f"fixed195-{row['language']}-s"):
            raise ValueError(f"Non-canonical single ID: {row.get('id')}")
        counts[row["language"]] += 1
        verify_audio(root, row)
    for language in LANGUAGE_ORDER:
        if counts[language] != singles_for(language):
            raise ValueError(f"Incomplete singles for {language}: {counts[language]}")
    return rows


def validate_public_multi(root: Path) -> list[dict]:
    path = root / "public_multi_unscored.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len({r.get("id") for r in rows}) != len(rows):
        raise ValueError("Duplicate multi case IDs")
    if len(rows) > 25:
        raise ValueError("Expected at most 25 public DataHive clips")
    for item in rows:
        if item.get("reference_status") != "missing_human_transcript":
            raise ValueError(f"Invalid public transcript status: {item.get('id')}")
        if item.get("is_scoreable_asr") is not False:
            raise ValueError(f"Public sample falsely marked scored: {item.get('id')}")
        if item.get("language") not in ("en", "hi", "id", "ru"):
            raise ValueError("Unexpected DataHive language")
        verify_audio(root, item)
        parts = item.get("isolated_audio") or []
        if len(parts) != 2 or {p.get("speaker") for p in parts} != {
            "speaker_1", "speaker_2"
        }:
            raise ValueError("Missing source speaker tracks")
        for speaker in parts:
            target = (root / "audio" / speaker["audio"]).resolve()
            if not target.is_relative_to(root.resolve()) or checksum(target) != speaker["sha256"]:
                raise ValueError("Changed/missing source speaker track")
    return rows


def verify_audio(root: Path, item: dict) -> Path:
    path = (root / item["audio"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Unsafe or missing audio for {item.get('id')}")
    if checksum(path) != item.get("sha256"):
        raise ValueError(f"Changed frozen audio for {item.get('id')}")
    return path


def fingerprint(items: list[dict], model: str, device: str, precision: str,
                mode: str) -> str:
    h = sha256()
    for value in (VERSION, model, device, precision, mode):
        h.update(value.encode("utf-8"))
    for item in items:
        for field in ("id", "sha256", "language", "reference", "source_revision",
                      "source_rev"):
            h.update(str(item.get(field, "")).encode("utf-8"))
            h.update(b"\0")
    base = Path(__file__).resolve().parents[1]
    for filename in CODE_PATHS:
        h.update(filename.encode())
        h.update((base / filename).read_bytes())
    return h.hexdigest()


def speech_transcript(prediction) -> tuple[str, list[dict]]:
    pieces = [
        {"text": u.text, "start_ms": u.start_ms, "end_ms": u.end_ms,
         "language": u.language, "timestamp_quality": u.timestamp_quality}
        for u in prediction.utterances
    ]
    return " ".join(x["text"] for x in pieces).strip(), pieces


def review_flags(text: str, utterances: list[dict], seconds: float) -> list[str]:
    """Low-confidence signals only, never a fluency or accuracy judgement."""
    flags = []
    if not text.strip():
        flags.append("empty_transcription")
    if seconds > 10 and len(text.strip()) < max(6, seconds * .35):
        flags.append("low_text_density_manual_review")
    if REPEAT_PATTERN.search(text):
        flags.append("possible_repeated_word")
    if any(len(x["text"]) > 320 for x in utterances):
        flags.append("long_utterance_check_segmentation")
    if any(x["end_ms"] <= x["start_ms"] for x in utterances):
        flags.append("invalid_timestamp")
    if any(utterances[i]["start_ms"] < utterances[i-1]["start_ms"]
           for i in range(1, len(utterances))):
        flags.append("nonmonotonic_utterance_order")
    return flags


def single_summary(cases: list[dict], rows: dict) -> dict:
    def aggregate(selection: list[dict]) -> dict:
        results = [rows[x["id"]] for x in selection if x["id"] in rows]
        good = [x for x in results if not x.get("error")]
        total = sum(x["reference_characters"] for x in good)
        return {
            "requested": len(selection), "completed": len(good),
            "failed": sum(bool(x.get("error")) for x in results),
            "not_run": len(selection) - len(results),
            "cer": (sum(x["character_edits"] for x in good) / total
                    if total else None),
            "cer_percent": (round(100 * sum(x["character_edits"] for x in good)
                                  / total, 3) if total else None),
            "mean_model_latency_ms": (round(sum(x["latency_ms"] for x in good)
                                              / len(good), 1) if good else None),
            "empty_predictions": sum(not x["hypothesis"] for x in good),
            "review_flags": sum(bool(x["review_flags"]) for x in good),
        }
    grouped = {
        language: aggregate([x for x in cases if x["language"] == language])
        for language in LANGUAGE_ORDER
    }
    rates = [x["cer"] for x in grouped.values() if x["cer"] is not None]
    old = aggregate([x for x in cases if x.get("reused_fixed150_id")])
    new = aggregate([x for x in cases if not x.get("reused_fixed150_id")])
    return {
        "overall": aggregate(cases),
        "per_language": grouped,
        "macro_cer": sum(rates)/len(rates) if len(rates) == len(LANGUAGE_ORDER)
                      else None,
        "old_56": old,
        "new_74": new,
        "scoring": "human-reference CER, not word intelligibility or meeting quality",
        "incomplete_results_not_counted_as_zero_error": True,
    }


def multi_summary(cases: list[dict], rows: dict) -> dict:
    good = [rows[x["id"]] for x in cases if x["id"] in rows
            and not rows[x["id"]].get("error")]
    return {
        "available_source_audio": len(cases),
        "reviewed_with_model": len(good),
        "manual_review_flags": sum(bool(x.get("review_flags")) for x in good),
        "without_human_reference": len(cases),
        "scoreable_cer": None,
        "scoreable_diarization": None,
        "human_intelligibility_rating": None,
        "note": "Only reviewable ASR transcripts. ASR cannot determine whether its own words "
                "match the source; verify completeness and speaker attribution by listening.",
    }


def run(root: Path, output: Path, mode: str, model_name: str,
        device: str, precision: str, max_cases: int | None = None) -> dict:
    from speech.whisper import WhisperEngine

    singles = validate_singles(root) if mode in ("singles", "all") else []
    multi = validate_public_multi(root) if mode in ("multi", "all") else []
    if mode == "multi" and not multi:
        raise FileNotFoundError(
            "No downloaded public authentic dialogues. "
            "Run tools/fixed195_prepare.py --source multi first."
        )
    all_cases = singles + multi
    fpr = fingerprint(all_cases, model_name, device, precision, mode)
    state = {
        "schema": VERSION, "fingerprint": fpr, "model": model_name,
        "mode": mode, "single_results": {}, "multi_results": {},
    }
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("schema") != VERSION or old.get("fingerprint") != fpr:
            raise ValueError("Old report fingerprint mismatch; use a different --output")
        state = old

    pending = [x for x in all_cases if (
        x["id"] not in state["single_results"] if x["kind"] == "single"
        else x["id"] not in state["multi_results"]
    )]
    if max_cases is not None:
        pending = pending[:max_cases]
    if pending:
        engine = WhisperEngine(model_name, device, precision)
        engine._get_model()
        for index, item in enumerate(pending, 1):
            target = state["single_results"] if item["kind"] == "single" else state["multi_results"]
            print(f"[{index}/{len(pending)}] {item['id']} "
                  f"{item['duration_s']:.1f}s", flush=True)
            try:
                samples = load_wav(verify_audio(root, item))
                started = time.perf_counter()
                result = engine.recognize(samples, language="auto")
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                hypothesis, segments = speech_transcript(result)
                flags = review_flags(hypothesis, segments, float(item["duration_s"]))
                common = {
                    "id": item["id"], "language": item["language"],
                    "duration_s": item["duration_s"], "latency_ms": latency_ms,
                    "hypothesis": hypothesis, "utterances": segments[:MAX_RECENT_UTTERANCES],
                    "review_flags": flags,
                    "predicted_language": result.language,
                    "speaker_attribution": "unmeasured",
                }
                if item["kind"] == "single":
                    score = score_pair(item["reference"], hypothesis,
                                       language=item["language"])
                    target[item["id"]] = {
                        **common,
                        "reference": item["reference"],
                        "character_edits": score["character_edits"],
                        "reference_characters": score["reference_characters"],
                        "cer": score["cer"],
                    }
                else:
                    target[item["id"]] = {
                        **common,
                        "reference_status": "unavailable",
                        "cer": None, "human_intelligibility_rating": None,
                        "verdict": "manual_listening_required",
                    }
            except Exception as exc:
                target[item["id"]] = {
                    "id": item["id"], "language": item["language"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            atomic_json(output, state)
            entry = target[item["id"]]
            if entry.get("error"):
                continue
            print(f"  CER={entry['cer']:.2%}" if item["kind"] == "single"
                  else f"  review_flags={entry['review_flags']}, "
                       "manual listening still required", flush=True)

    summary = {
        "schema": VERSION + "-summary",
        "mode": mode,
        "singles": single_summary(singles, state["single_results"]) if singles else None,
        "multi": multi_summary(multi, state["multi_results"]) if mode != "singles" else None,
        "human_reference_available_for_multi": False,
        "completed_model_testing_is_not_manual_fluency_validation": True,
    }
    atomic_json(output.with_name(output.stem + "_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--mode", choices=("singles", "multi", "all"), default="singles")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cases", type=int, help="Optional smoke-test limit; resume to finish")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be >=1")
    output = args.output or args.root / f"content_review_{args.mode}.json"
    run(args.root, output, args.mode, args.model, args.device, args.compute_type,
        args.max_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired full-corpus test of strict versus evidence-preserving ASR verification.

One Whisper decode per acoustic window; the exact *same* segment objects are
scored as raw, strict-verified, and preserved-verified outputs. No forced
reference language, oracle switching, or added model calls for the comparison.
All reference transcripts and frozen audio remain unmodified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from speech.audio_windows import local_speech_regions, speech_windows
from speech.quality_guard import verify_utterances
from speech.utterance_units import split_utterances
from tools.fixed150_prepare import atomic_json, verify_lock
from tools.fixed150_reference_audit import FOCUS_15, source_flags
from tools.phase07_benchmark import load_wav

VERSION = "meeting-fixed150-guard-audit-v1"
RELATED_CODE = (
    "speech/whisper.py", "speech/audio_windows.py",
    "speech/multilingual_windows.py", "speech/quality_guard.py",
    "speech/utterance_units.py", "speech/text_format.py",
    "phase07/quality.py", "tools/fixed150_guard_audit.py",
)


def fingerprint(lock: dict, model: str, scope: str) -> str:
    repo = Path(__file__).resolve().parents[1]
    h = sha256()
    for token in (VERSION, model, scope):
        h.update(token.encode())
    for case in lock["cases"]:
        h.update(case["id"].encode())
        h.update(case["sha256"].encode())
    for relative in RELATED_CODE:
        h.update(relative.encode())
        h.update((repo / relative).read_bytes())
    return h.hexdigest()


def metric(reference: str, text: str, language: str) -> dict:
    score = score_pair(reference, text, language=language)
    return {
        "cer": score["cer"],
        "character_edits": score["character_edits"],
        "reference_characters": score["reference_characters"],
        "output_characters": len(text),
    }


def make_kwargs(options) -> dict:
    return {
        "language": None,
        "multilingual": True,
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {
            "min_speech_duration_ms": options.min_speech_duration_ms,
            "min_silence_duration_ms": options.min_silence_duration_ms,
            "speech_pad_ms": options.speech_pad_ms,
        },
        "no_speech_threshold": .5,
        "log_prob_threshold": -1.,
        "compression_ratio_threshold": 2.4,
        "hallucination_silence_threshold": 1.2,
        "condition_on_previous_text": False,
        "temperature": 0,
    }


def assess_one(model, root: Path, case: dict) -> dict:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    samples = load_wav(root / case["audio"])
    options = VadOptions(
        min_speech_duration_ms=200, min_silence_duration_ms=650,
        speech_pad_ms=180,
    )
    regions = get_speech_timestamps(samples, options, sampling_rate=16000)
    windows = speech_windows(regions, len(samples))
    if len(windows) > 100:
        raise RuntimeError("Unexpected number of windows")
    kwargs = make_kwargs(options)
    raw_fragments: list[str] = []
    strict_segments = []
    preserved_segments = []
    window_languages = []
    total_model_segments = 0
    segment_fallbacks = 0
    segment_rejections_strict = 0
    segment_rejections_preserved = 0
    started = time.perf_counter()
    for start, end in windows:
        local = local_speech_regions(regions, start, end)
        if not local:
            continue
        model_segments, info = model.transcribe(samples[start:end], **kwargs)
        # IMPORTANT: consume once, then verify the identical emitted objects.
        pieces = list(model_segments)
        language = str(getattr(info, "language", "und") or "und").lower()
        prob = getattr(info, "language_probability", None)
        if prob is not None and float(prob) < .5:
            language = "und"
        window_languages.append(language)
        total_model_segments += len(pieces)
        raw_fragments.extend(
            str(getattr(p, "text", "") or "").strip() for p in pieces
            if str(getattr(p, "text", "") or "").strip()
        )
        duration_ms = round((end - start) / 16)
        strict = verify_utterances(pieces, local, duration_ms, language)
        preserved = verify_utterances(
            pieces, local, duration_ms, language,
            preserve_disputed_text=True,
        )
        segment_fallbacks += sum(
            u.timestamp_quality == "segment" for u in preserved
        ) - sum(u.timestamp_quality == "segment" for u in strict)
        # Rejections are an upper-bound proxy: one model segment may split into
        # several valid utterances, so these are deliberately not labeled "lost words".
        segment_rejections_strict += int(bool(pieces) and not strict)
        segment_rejections_preserved += int(bool(pieces) and not preserved)
        offset_ms = round(start / 16)
        # Only text and source segment time are needed here; use real utterance
        # objects for the splitter to match the application's reconstruction.
        from speech.contracts import Utterance, Word
        for output, collection in ((strict, strict_segments), (preserved, preserved_segments)):
            for u in output:
                words = tuple(Word(w.text, w.start_ms + offset_ms,
                                   w.end_ms + offset_ms) for w in u.words)
                collection.append(Utterance(
                    u.text, u.start_ms + offset_ms, u.end_ms + offset_ms,
                    language, words, u.timestamp_quality, u.confidence,
                    language_confidence=prob,
                ))
    strict_segments.sort(key=lambda x: (x.start_ms, x.end_ms))
    preserved_segments.sort(key=lambda x: (x.start_ms, x.end_ms))
    strict_text = " ".join(
        s.text for s in split_utterances(tuple(strict_segments))
    )
    preserved_text = " ".join(
        s.text for s in split_utterances(tuple(preserved_segments))
    )
    raw_text = " ".join(raw_fragments)
    reference = case["reference"]
    return {
        "id": case["id"], "sha256": case["sha256"],
        "language": case["language"], "locale": case.get("locale"),
        "duration_s": case["duration_s"],
        "reference_review_flags": source_flags(case["duration_s"], reference),
        "windows": len(windows), "model_segments": total_model_segments,
        "segment_timestamp_fallbacks_extra": segment_fallbacks,
        "windows_with_all_model_output_rejected_strict": segment_rejections_strict,
        "windows_with_all_model_output_rejected_preserved": segment_rejections_preserved,
        "detected_languages": sorted(set(window_languages)),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "window_raw": metric(reference, raw_text, case["language"]),
        "strict": metric(reference, strict_text, case["language"]),
        "preserved": metric(reference, preserved_text, case["language"]),
        "preserved_minus_strict_pp": round(
            (score_pair(reference, preserved_text, language=case["language"])["cer"]
             - score_pair(reference, strict_text, language=case["language"])["cer"]) * 100, 3
        ),
    }


def summarize(rows: list[dict], total: int) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)

    def group(items: list[dict]) -> dict:
        denominator = sum(x["strict"]["reference_characters"] for x in items)
        if not denominator:
            return {"samples": len(items), "weighted_cer": None}
        cer = {
            name: sum(x[name]["character_edits"] for x in items) / denominator
            for name in ("window_raw", "strict", "preserved")
        }
        return {
            "samples": len(items), "weighted_cer": cer,
            "improved": sum(x["preserved_minus_strict_pp"] < -0.0005 for x in items),
            "worsened": sum(x["preserved_minus_strict_pp"] > 0.0005 for x in items),
            "unchanged": sum(abs(x["preserved_minus_strict_pp"]) <= 0.0005 for x in items),
            "worsened_2pp_or_more": sum(
                x["preserved_minus_strict_pp"] >= 2 for x in items
            ),
            "improved_2pp_or_more": sum(
                x["preserved_minus_strict_pp"] <= -2 for x in items
            ),
            "review_reference_count": sum(
                bool(x["reference_review_flags"]) for x in items
            ),
        }

    all_languages = set(FOCUS_15)
    focused = [r for r in rows if r["language"] in all_languages]
    without_flagged = [r for r in rows if not r["reference_review_flags"]]
    return {
        "complete": len(rows) == total,
        "completed": len(rows), "requested": total,
        "overall": group(rows),
        "priority_15": group(focused),
        "reference_review_sensitivity_only": group(without_flagged),
        "per_language": {
            name: group(items) for name, items in sorted(groups.items())
        },
        "top_10_worsened": [
            {"id": r["id"], "language": r["language"],
             "preserved_minus_strict_pp": r["preserved_minus_strict_pp"]}
            for r in sorted(rows,
                            key=lambda row: row["preserved_minus_strict_pp"],
                            reverse=True)[:10]
        ],
        "top_10_improved": [
            {"id": r["id"], "language": r["language"],
             "preserved_minus_strict_pp": r["preserved_minus_strict_pp"]}
            for r in sorted(rows,
                            key=lambda row: row["preserved_minus_strict_pp"])[:10]
        ],
        "release_certified": False,
        "methodology": [
            "Exactly one model decode per window; three scoring policies reuse the same output.",
            "Do not use source-reference CER to choose between strict/preserved in production.",
            "Original frozen labels remain unchanged and suspected references remain in raw scores.",
            "This test is single-speaker and cannot certify code switching or overlapping speech.",
            "Regression candidates need manual acoustic verification and independent holdout samples.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--scope", choices=("all", "focus"), default="all")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    root = args.root
    lock = verify_lock(root)
    all_singles = [c for c in lock["cases"] if c["kind"] == "single"]
    if len(all_singles) != 100:
        raise ValueError("Expected exactly 100 frozen single-speaker cases")
    selected = [
        c for c in all_singles if args.scope == "all"
        or c["language"] in set(FOCUS_15)
    ]
    expected = 100 if args.scope == "all" else 68
    if len(selected) != expected:
        raise ValueError("Frozen source allocation changed")
    destination = args.output or root / ("guard_audit_" + args.scope + ".json")
    fp = fingerprint(lock, args.model, args.scope)
    data = {"schema": VERSION, "fingerprint": fp, "model": args.model,
            "scope": args.scope, "cases": {}}
    if destination.exists():
        data = json.loads(destination.read_text(encoding="utf-8"))
        if (data.get("schema") != VERSION
            or data.get("fingerprint") != fp
            or data.get("scope") != args.scope):
            raise ValueError("Incompatible checkpoint; choose a new --output path")
    for case in selected:
        prior = data["cases"].get(case["id"])
        if prior and prior["sha256"] != case["sha256"]:
            raise ValueError(f"Unexpected changed source for {case['id']}")
    pending = [c for c in selected if c["id"] not in data["cases"]]
    if args.limit:
        pending = pending[:args.limit]
    if pending:
        from speech.whisper import WhisperEngine
        model = WhisperEngine(args.model, "cuda", "float16")._get_model()
        for case in pending:
            print(f"[{len(data['cases']) + 1}/{len(selected)}] "
                  f"{case['id']} {case['language']} {case['duration_s']:.1f}s",
                  flush=True)
            row = assess_one(model, root, case)
            data["cases"][case["id"]] = row
            atomic_json(destination, data)
            print(
                f"  raw={row['window_raw']['cer']:.2%} "
                f"strict={row['strict']['cer']:.2%} "
                f"preserved={row['preserved']['cer']:.2%} "
                f"delta={row['preserved_minus_strict_pp']:+.2f}pp",
                flush=True,
            )
    measured = [
        data["cases"][c["id"]] for c in selected if c["id"] in data["cases"]
    ]
    results = summarize(measured, expected)
    summary_path = destination.with_name(destination.stem + "_summary.json")
    atomic_json(summary_path, {
        "schema": VERSION + "-summary", "scope": args.scope,
        "fingerprint": fp, **results,
    })
    print("Status:", "COMPLETE" if results["complete"] else "PARTIAL",
          f"{results['completed']}/{results['requested']}")
    for label in ("overall", "priority_15"):
        record = results[label]
        rates = record.get("weighted_cer")
        if isinstance(rates, dict):
            print(label,
                  "strict", f"{rates['strict']:.2%}",
                  "preserved", f"{rates['preserved']:.2%}",
                  "improved/worsened/unchanged",
                  record["improved"], record["worsened"], record["unchanged"])
    print("Top worsened:", results["top_10_worsened"])
    print("Checkpoint:", destination)
    print("Summary:", summary_path)
    print("Production settings unchanged. Scores do not certify a release.")
    return 0 if results["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

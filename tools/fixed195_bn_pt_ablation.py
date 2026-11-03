"""Controlled Whisper ablation for Bengali and Portuguese Fixed195 singles.

Baseline uses the existing 130-case auto windowed production run. Compare:
  whole_auto_verified    Same WhisperEngine with per-window LID disabled.
  whole_auto_raw         Whole-clip model text before the acoustic word guard.
  whole_forced_verified  Same WhisperEngine with the KNOWN language forced.
  whole_forced_raw       Whole-clip model text with the KNOWN language forced.

Known-language branches are ORACLE-CONDITIONED diagnostics because language
comes from the gold manifest. They only represent a deployable user-selected
language workflow, NOT automatic language detection in live meetings.
All 20 clips are kept in results, including regressions. No changes to
production defaults and no use of reference text in candidate decoding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from tools.fixed150_prepare import atomic_json
from tools.fixed150_language_anchor import params
from tools.fixed195_content_review import (
    fingerprint as baseline_fingerprint,
    load_manifest, speech_transcript, verify_audio,
)
from tools.phase07_benchmark import load_wav

VERSION = "meeting-fixed195-bn-pt-ablation-v1"
VARIANTS = (
    "whole_auto_verified",
    "whole_auto_raw",
    "whole_forced_verified",
    "whole_forced_raw",
)
RELATED = (
    "tools/fixed195_bn_pt_ablation.py",
    "tools/fixed195_content_review.py",
    "tools/fixed150_language_anchor.py",
    "speech/whisper.py",
    "speech/multilingual_windows.py",
    "speech/quality_guard.py",
    "speech/audio_windows.py",
    "speech/utterance_units.py",
    "speech/text_format.py",
    "phase07/quality.py",
)


@contextmanager
def override_window_lid(value: str):
    name = "MEETING_WHISPER_WINDOW_LID"
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def verified_candidate(engine, audio, language: str, variant: str) -> dict:
    if variant == "whole_auto_verified":
        with override_window_lid("0"):
            prediction = engine.recognize(audio, language="auto")
    elif variant == "whole_forced_verified":
        prediction = engine.recognize(audio, language=language)
    else:
        raise ValueError(f"Not a verified variant: {variant}")
    hypothesis, utterances = speech_transcript(prediction)
    return {
        "hypothesis": hypothesis,
        "reported_language": prediction.language,
        "segments": len(utterances),
        "timestamp_source": "production_verified",
        "invalid_replacement_chars": hypothesis.count("\ufffd"),
    }


def raw_candidate(engine, audio, language: str, variant: str) -> dict:
    from faster_whisper.vad import VadOptions

    if variant not in ("whole_auto_raw", "whole_forced_raw"):
        raise ValueError(f"Not a raw variant: {variant}")
    forced = variant == "whole_forced_raw"
    options = VadOptions(
        min_speech_duration_ms=200, min_silence_duration_ms=650,
        speech_pad_ms=180,
    )
    # Keep the same decoding/VAD options as the production Whisper adapter.
    kwargs = params(options, language=language if forced else None)
    pieces, info = engine._get_model().transcribe(audio, **kwargs)
    segments = list(pieces)
    hypothesis = " ".join(
        str(getattr(segment, "text", "") or "").strip()
        for segment in segments
        if str(getattr(segment, "text", "") or "").strip()
    ).strip()
    return {
        "hypothesis": hypothesis,
        "reported_language": (language if forced else
                              str(getattr(info, "language", "und") or "und")),
        "segments": len(segments),
        "timestamp_source": "raw_model_segments_unverified",
        "invalid_replacement_chars": hypothesis.count("\ufffd"),
    }


def reference_case_check(manifest: dict, baseline: dict) -> None:
    if baseline.get("id") != manifest.get("id") or (
            baseline.get("language") != manifest.get("language")):
        raise ValueError(f"Baseline case ID/language mismatch: {manifest.get('id')}")
    if baseline.get("reference") != manifest.get("reference"):
        raise ValueError(f"Baseline reference changed: {manifest.get('id')}")
    actual = score_pair(
        manifest["reference"], baseline.get("hypothesis", ""),
        language=manifest["language"],
    )
    if (actual["character_edits"] != baseline.get("character_edits")
            or actual["reference_characters"] !=
            baseline.get("reference_characters")):
        raise ValueError(f"Baseline CER is inconsistent: {manifest['id']}")


def case_score(case: dict, candidate: dict) -> dict:
    score = score_pair(case["reference"], candidate["hypothesis"],
                       language=case["language"])
    return {
        **candidate,
        "edits": score["character_edits"],
        "reference_chars": score["reference_characters"],
        "cer": score["cer"],
    }


def weighted(rows: list[dict]) -> dict:
    ref = sum(r["reference_chars"] for r in rows)
    edits = sum(r["edits"] for r in rows)
    return {"samples": len(rows), "edits": edits, "reference_chars": ref,
            "cer": edits/ref if ref else None,
            "cer_percent": round(100 * edits/ref, 3) if ref else None}


def summarize(items: list[dict], variants: tuple[str, ...] = VARIANTS) -> dict:
    groups = defaultdict(list)
    for item in items:
        groups[item["language"]].append(item)

    def group(rows: list[dict]) -> dict:
        baseline_rows = [r["baseline"] for r in rows]
        baseline = weighted(baseline_rows)
        candidate_scores = {}
        for variant in variants:
            done = [r for r in rows if variant in r["candidates"]
                    and not r["candidates"][variant].get("error")]
            failed = [r for r in rows if variant in r["candidates"]
                      and r["candidates"][variant].get("error")]
            tested = [r["candidates"][variant] for r in done]
            better = sum(r["candidates"][variant]["edits"] < r["baseline"]["edits"]
                         for r in done)
            worse = sum(r["candidates"][variant]["edits"] > r["baseline"]["edits"]
                        for r in done)
            same = len(done) - better - worse
            score = weighted(tested)
            score.update({
                "requested": len(rows),
                "not_run": len(rows) - len(done) - len(failed),
                "failed": len(failed),
                "improved_cases": better,
                "worsened_cases": worse,
                "unchanged_cases": same,
                "delta_pp": (round((score["cer"] - baseline["cer"]) * 100, 3)
                             if score["cer"] is not None
                             and len(done) == len(rows) else None),
                "relative_cer_comparable": len(done) == len(rows),
                "mean_decode_ms": (round(sum(r["candidates"][variant]["decode_ms"]
                                             for r in done)/len(done), 1)
                                   if done else None),
                "replacement_char_cases": sum(
                    r["candidates"][variant]["invalid_replacement_chars"] > 0
                    for r in done
                ),
            })
            candidate_scores[variant] = score
        return {"baseline": baseline, "candidates": candidate_scores}
    return {
        "overall": group(items),
        "per_language": {k: group(groups[k]) for k in sorted(groups)},
        "oracle_warning": (
            "ORACLE-CONDITIONED: forced branches use the known manifest language. "
            "only if the language is explicitly selected before decoding, "
            "not as automatic language identification."
        ),
        "raw_warning": (
            "Raw candidates bypass the acoustic word guard; a lower CER "
            "does not imply reliable timestamps or safe production output."
        ),
        "deployment_approved": False,
    }


def ablation_fingerprint(cases: list[dict], baseline_state: dict,
                         model: str, device: str, precision: str,
                         variants: tuple[str, ...]) -> str:
    h = sha256()
    for value in (VERSION, model, device, precision, "|".join(variants),
                  baseline_state["fingerprint"]):
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    for case in cases:
        for field in ("id", "sha256", "language", "reference"):
            h.update(str(case[field]).encode("utf-8"))
            h.update(b"\0")
    base = Path(__file__).resolve().parents[1]
    for source in RELATED:
        h.update(source.encode("utf-8"))
        h.update((base / source).read_bytes())
    return h.hexdigest()


def run(root: Path, baseline_path: Path, output: Path, model_id: str,
        device: str, precision: str, variants: tuple[str, ...],
        max_cases: int | None) -> dict:
    from speech.whisper import WhisperEngine

    all_cases = load_manifest(root / "manifest_single.jsonl", 130)
    cases = [x for x in all_cases if x["language"] in ("bn", "pt")]
    if (len(cases) != 20 or sum(x["language"] == "bn" for x in cases) != 10
            or sum(x["language"] == "pt" for x in cases) != 10):
        raise ValueError("Expected 10 Bengali and 10 Portuguese single samples")
    for case in cases:
        verify_audio(root, case)

    state_before = json.loads(baseline_path.read_text(encoding="utf-8"))
    if state_before.get("mode") != "singles" or state_before.get("model") != model_id:
        raise ValueError("Baseline must be the existing 130-case Whisper singles")
    expected = baseline_fingerprint(all_cases, model_id, device,
                                    precision, "singles")
    if state_before.get("fingerprint") != expected:
        raise ValueError(
            "Baseline fingerprint differs from model/source/code settings. "
            "Use the exact same baseline run or a new separate comparison."
        )

    original_results = state_before.get("single_results", {})
    if len(original_results) != 130 or any(x.get("error") for x in
                                           original_results.values()):
        raise ValueError("Baseline must have all 130 completed without failures")
    for case in cases:
        reference_case_check(case, original_results[case["id"]])

    fpr = ablation_fingerprint(cases, state_before, model_id, device,
                              precision, variants)
    checkpoint = {
        "schema": VERSION, "fingerprint": fpr,
        "model": model_id, "device": device, "precision": precision,
        "variants": list(variants), "cases": {},
    }
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("schema") != VERSION or previous.get("fingerprint") != fpr:
            raise ValueError("Checkpoint mismatch; choose another --output")
        checkpoint = previous

    for case in cases:
        cid = case["id"]
        original = original_results[cid]
        old = checkpoint["cases"].get(cid)
        if old is None:
            checkpoint["cases"][cid] = {
                "id": cid, "language": case["language"],
                "sha256": case["sha256"],
                "source_group": ("old" if case.get("reused_fixed150_id") else "new"),
                "baseline": {
                    "hypothesis": original["hypothesis"],
                    "edits": original["character_edits"],
                    "reference_chars": original["reference_characters"],
                    "cer": original["cer"],
                },
                "candidates": {},
            }
        elif old["sha256"] != case["sha256"]:
            raise ValueError(f"Source case fingerprint mismatch {cid}")
    atomic_json(output, checkpoint)

    pending = [(case, variant) for case in cases for variant in variants
               if variant not in checkpoint["cases"][case["id"]]["candidates"]]
    if max_cases is not None:
        todo_ids = list(dict.fromkeys(x["id"] for x, _ in pending))[:max_cases]
        pending = [(case, variant) for case, variant in pending
                   if case["id"] in todo_ids]
    if pending:
        engine = WhisperEngine(model_id, device, precision)
        engine._get_model()
        for index, (case, variant) in enumerate(pending, 1):
            cid = case["id"]
            print(f"[{index}/{len(pending)}] {cid} {variant} "
                  f"({case['duration_s']:.1f}s)", flush=True)
            started = time.perf_counter()
            try:
                audio = load_wav(verify_audio(root, case))
                if variant.endswith("_verified"):
                    candidate = verified_candidate(engine, audio, case["language"], variant)
                else:
                    candidate = raw_candidate(engine, audio, case["language"], variant)
                candidate["decode_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2)
                scored = case_score(case, candidate)
                checkpoint["cases"][cid]["candidates"][variant] = scored
                print(f"  CER={scored['cer']:.2%} "
                      f"baseline={checkpoint['cases'][cid]['baseline']['cer']:.2%} "
                      f"language={scored['reported_language']} "
                      f"U+FFFD={scored['invalid_replacement_chars']}", flush=True)
            except Exception as exc:
                checkpoint["cases"][cid]["candidates"][variant] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
                print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            atomic_json(output, checkpoint)

    items = [checkpoint["cases"][x["id"]] for x in cases]
    summary = {
        "schema": VERSION + "-summary",
        "total_cases": len(cases),
        "variants": list(variants),
        **summarize(items, variants),
    }
    destination = output.with_name(output.stem + "_summary.json")
    atomic_json(destination, summary)
    for language in ("bn", "pt"):
        part = summary["per_language"][language]
        print(f"{language} baseline={part['baseline']['cer_percent']}%")
        for variant, result in part["candidates"].items():
            print(f"  {variant}: {result['cer_percent']}% "
                  f"delta={result['delta_pp']}pp "
                  f"better={result['improved_cases']} "
                  f"worse={result['worsened_cases']} "
                  f"failed={result['failed']}")
    print(f"Checkpoint: {output}")
    print(f"Summary: {destination}")
    print("Offline controlled diagnostic only, no production behavior changed.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be >= 1")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must be distinct")
    result = args.output or args.root / "bn_pt_ablation.json"
    if args.baseline and result.resolve() == args.baseline.resolve():
        parser.error("Do not overwrite baseline results")
    run(args.root,
        args.baseline or args.root / "content_review_singles.json",
        result, args.model, args.device, args.compute_type,
        tuple(args.variants), args.max_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

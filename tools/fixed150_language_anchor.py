"""Offline diagnostic for low-confidence language drift in segmented ASR.

The candidate language anchor is derived exclusively from independently
high-confidence model language detections, NEVER from ground-truth labels.
This file is an offline, retrospective A/B comparison. Future windows may
inform the anchor; consequently it MUST NOT be used as a streaming or
production decoder without a causal, latency-bounded reimplementation and
real code-switch validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair
from speech.audio_windows import speech_windows, local_speech_regions
from speech.quality_guard import verify_utterances
from speech.utterance_units import split_utterances
from speech.contracts import Utterance, Word
from tools.fixed150_prepare import atomic_json, verify_lock
from tools.fixed150_reference_audit import source_flags
from tools.phase07_benchmark import load_wav

VERSION = "meeting-fixed150-lid-anchor-v1"
RELATED = (
    "speech/audio_windows.py", "speech/quality_guard.py",
    "speech/utterance_units.py", "speech/text_format.py",
    "speech/whisper.py", "tools/fixed150_language_anchor.py",
)


def stable_probability(value: object) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return p if math.isfinite(p) and 0 <= p <= 1 else None


def infer_anchor(
    evidence: list[dict], *, min_probability: float = .90,
    min_witnesses: int = 2, min_speech_seconds: float = 3.0,
) -> str | None:
    """Choose a stable, non-oracle anchor only with independent strong votes.

    Requiring agreement on >=2 distinct windows avoids a single confident
    hallucination overriding legitimate language switches. A close competitor
    or multiple credible languages rejects the anchor entirely.
    """
    strong = [
        item for item in evidence
        if (item.get("language") not in (None, "", "auto", "und", "mul")
            and (p := stable_probability(item.get("probability"))) is not None
            and p >= min_probability
            and float(item.get("speech_seconds", 0)) > 0)
    ]
    vote_counts = Counter(x["language"] for x in strong)
    if not vote_counts:
        return None
    # Treat conflicting high-confidence languages as genuine possible
    # code-switching; no monolingual assumption is safe.
    if len(vote_counts) != 1:
        return None
    candidate, votes = next(iter(vote_counts.items()))
    if votes < min_witnesses:
        return None
    duration = sum(float(x["speech_seconds"]) for x in strong)
    return candidate if duration >= min_speech_seconds else None


def needs_redecode(item: dict, anchor: str | None, *,
                   uncertain_below: float = .70) -> bool:
    """A high-confidence alternative is NEVER forcibly replaced."""
    if not anchor or item.get("language") == anchor:
        return False
    p = stable_probability(item.get("probability"))
    if p is None or p >= uncertain_below:
        return False
    # Silence-like windows often return no output and should not be forced
    # into producing words. Skip these until stronger acoustic evidence.
    if not item.get("has_model_text"):
        return False
    return float(item.get("speech_seconds", 0)) >= .25


def params(vad_options, language: str | None = None) -> dict:
    return {
        "language": language,
        # A forced anchor must not be overridden by another per-segment LID.
        "multilingual": language is None,
        "beam_size": 5, "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {
            "min_speech_duration_ms": vad_options.min_speech_duration_ms,
            "min_silence_duration_ms": vad_options.min_silence_duration_ms,
            "speech_pad_ms": vad_options.speech_pad_ms,
        },
        "no_speech_threshold": .5,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "hallucination_silence_threshold": 1.2,
        "condition_on_previous_text": False, "temperature": 0,
    }


def decode_and_verify(model, samples, kwargs: dict,
                      speech_regions: list[dict], start: int,
                      end: int) -> dict:
    started = time.perf_counter()
    pieces, info = model.transcribe(samples[start:end], **kwargs)
    pieces = list(pieces)
    spent_ms = round((time.perf_counter() - started) * 1000, 2)
    language = str(getattr(info, "language", "und") or "und").lower()
    probability = stable_probability(getattr(info, "language_probability", None))
    text_detected = any(str(getattr(p, "text", "") or "").strip() for p in pieces)
    local = local_speech_regions(speech_regions, start, end)
    duration_ms = round((end - start) / 16)
    actual_language = (language if probability is None or probability >= .50
                       else "und")
    checked = verify_utterances(pieces, local, duration_ms, actual_language)
    offset_ms = round(start / 16)
    words = [
        Utterance(
            u.text, u.start_ms + offset_ms, u.end_ms + offset_ms,
            u.language, tuple(
                Word(w.text, w.start_ms + offset_ms, w.end_ms + offset_ms)
                for w in u.words
            ), u.timestamp_quality, u.confidence,
            language_confidence=probability,
        ) for u in checked
    ]
    words.sort(key=lambda u: (u.start_ms, u.end_ms))
    output = " ".join(x.text for x in split_utterances(tuple(words)))
    return {
        "text": output, "language": language,
        "probability": probability, "has_model_text": text_detected,
        "segments_raw": len(pieces), "segments_kept": len(words),
        "latency_ms": spent_ms,
    }


def evaluate_case(model, root: Path, case: dict) -> dict:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = load_wav(root / case["audio"])
    opts = VadOptions(
        min_speech_duration_ms=200, min_silence_duration_ms=650,
        speech_pad_ms=180,
    )
    regions = get_speech_timestamps(audio, opts, sampling_rate=16000)
    windows = speech_windows(regions, len(audio))
    if len(windows) > 100:
        raise RuntimeError("Unexpected number of speech windows")
    baseline = []
    for start, end in windows:
        local = local_speech_regions(regions, start, end)
        if not local:
            continue
        result = decode_and_verify(model, audio, params(opts), regions, start, end)
        result.update({
            "start": start, "end": end,
            "speech_seconds": round(sum(max(
                0, item["end"] - item["start"]) for item in local) / 16000, 3),
        })
        baseline.append(result)
    # WARNING: this retrospective anchor uses future independent evidence.
    # A causal implementation needs delayed commitments and explicit latency.
    anchor = infer_anchor(baseline)
    redecoded = []
    for item in baseline:
        if needs_redecode(item, anchor):
            forced = decode_and_verify(
                model, audio, params(opts, anchor), regions,
                item["start"], item["end"],
            )
            redecoded.append({**item, "replacement": forced["text"],
                              "retry_latency_ms": forced["latency_ms"]})
        else:
            redecoded.append({**item, "replacement": item["text"],
                              "retry_latency_ms": 0})
    original_text = " ".join(x["text"] for x in baseline if x["text"])
    candidate_text = " ".join(x["replacement"] for x in redecoded
                              if x["replacement"])
    baseline_score = score_pair(case["reference"], original_text,
                                language=case["language"])
    candidate_score = score_pair(case["reference"], candidate_text,
                                 language=case["language"])
    return {
        "id": case["id"], "sha256": case["sha256"],
        "language": case["language"],
        "duration_s": case["duration_s"],
        "reference_review_flags": source_flags(
            case["duration_s"], case["reference"]),
        "anchor_from_model_only": anchor,
        "auto_detected_languages": [
            {"start_ms": round(x["start"] / 16),
             "end_ms": round(x["end"] / 16),
             "code": x["language"],
             "probability": x["probability"],
             "has_model_text": x["has_model_text"],
             "retried": x["retry_latency_ms"] > 0}
            for x in redecoded
        ],
        "retry_count": sum(x["retry_latency_ms"] > 0 for x in redecoded),
        "auto_decode_ms": round(sum(x["latency_ms"] for x in baseline), 2),
        "retry_decode_ms": round(sum(x["retry_latency_ms"] for x in redecoded), 2),
        "baseline": {
            "cer": baseline_score["cer"],
            "edits": baseline_score["character_edits"],
            "reference_chars": baseline_score["reference_characters"],
        },
        "candidate": {
            "cer": candidate_score["cer"],
            "edits": candidate_score["character_edits"],
            "reference_chars": candidate_score["reference_characters"],
        },
        "delta_pp": round(100 * (
            candidate_score["cer"] - baseline_score["cer"]), 3),
    }


def fingerprint(lock: dict, model_id: str, chosen: list[dict]) -> str:
    h = sha256()
    repo = Path(__file__).resolve().parents[1]
    for item in (VERSION, model_id):
        h.update(item.encode())
    for case in chosen:
        h.update(case["id"].encode())
        h.update(case["sha256"].encode())
    for path in RELATED:
        h.update(path.encode())
        h.update((repo / path).read_bytes())
    return h.hexdigest()


def run(root: Path, model_id: str, ids: list[str], output: Path) -> dict:
    from speech.whisper import WhisperEngine

    lock = verify_lock(root)
    singles = {x["id"]: x for x in lock["cases"] if x["kind"] == "single"}
    if len(singles) != 100:
        raise ValueError("Require the original frozen 100 single-speaker cases")
    missing = set(ids) - set(singles)
    if missing:
        raise ValueError(f"Unknown case IDs {sorted(missing)}")
    chosen = [singles[x] for x in ids]
    fpr = fingerprint(lock, model_id, chosen)
    state = {"schema": VERSION, "fingerprint": fpr,
             "model": model_id, "cases": {}}
    if output.exists():
        state = json.loads(output.read_text(encoding="utf-8"))
        if (state.get("schema") != VERSION or
            state.get("fingerprint") != fpr):
            raise ValueError("Checkpoint does not match source/model/code")
    pending = [x for x in chosen if x["id"] not in state["cases"]]
    if pending:
        engine = WhisperEngine(model_id, "cuda", "float16")
        model = engine._get_model()
        for case in pending:
            print(f"[{len(state['cases'])+1}/{len(ids)}] "
                  f"{case['id']} {case['language']} {case['duration_s']:.1f}s",
                  flush=True)
            row = evaluate_case(model, root, case)
            state["cases"][case["id"]] = row
            atomic_json(output, state)
            print(
                f"  anchor={row['anchor_from_model_only']} "
                f"retries={row['retry_count']} "
                f"baseline={row['baseline']['cer']:.2%} "
                f"candidate={row['candidate']['cer']:.2%} "
                f"change={row['delta_pp']:+.2f}pp",
                flush=True,
            )
    items = [state["cases"][x] for x in ids]
    den = sum(x["baseline"]["reference_chars"] for x in items)
    summary = {
        "schema": VERSION + "-summary",
        "total": len(items),
        "anchor_cases": sum(x["anchor_from_model_only"] is not None
                            for x in items),
        "retried_windows": sum(x["retry_count"] for x in items),
        "baseline_cer": (sum(x["baseline"]["edits"] for x in items) / den
                         if den else None),
        "candidate_cer": (sum(x["candidate"]["edits"] for x in items) / den
                          if den else None),
        "improved_cases": sum(x["delta_pp"] < -.0005 for x in items),
        "worsened_cases": sum(x["delta_pp"] > .0005 for x in items),
        "unchanged_cases": sum(abs(x["delta_pp"]) <= .0005 for x in items),
        "no_reference_oracle": True,
        "retrospective_noncausal_anchor": True,
        "release_certified": False,
    }
    target = output.with_name(output.stem + "_summary.json")
    atomic_json(target, summary)
    print("Summary:", summary)
    print("Checkpoint:", output)
    print("Summary file:", target)
    print("IMPORTANT: offline retrospective comparison; NOT production code.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--case-ids", nargs="+", default=[
        "fixed150-048", "fixed150-047", "fixed150-045",
        "fixed150-097", "fixed150-043", "fixed150-071",
        "fixed150-069", "fixed150-072",
    ])
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/fixed150/language_anchor_audit.json"))
    args = parser.parse_args()
    if not args.case_ids or len(set(args.case_ids)) != len(args.case_ids):
        parser.error("Case IDs must be unique and nonempty")
    run(args.root, args.model, args.case_ids, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

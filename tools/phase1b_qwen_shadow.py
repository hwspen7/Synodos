"""Bounded, opt-in Qwen3-ASR shadow review of VAD-supported caption gaps.

Reads ONLY microphone audio, independent VAD and final saved captions.
AMI ground-truth words and speaker turn reference are never used to choose
windows. Candidate transcripts are private hypotheses, never live captions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from speech.contracts import Utterance
from speech.gap_recovery import uncovered_voice_intervals
from speech.qwen import qwen_language_name
from tools.ami_shadow10_prepare import checksum
from tools.phase07_benchmark import load_wav

SCHEMA = "meeting-phase1b-qwen-asr-shadow-v1"
MODEL = "Qwen/Qwen3-ASR-1.7B"


def select_gaps(vad_regions: list[dict], final_segments: list[dict],
                duration_ms: int, *, min_ms: int = 900,
                max_windows: int = 6, max_total_ms: int = 60000) -> list[dict]:
    """Select bounded windows using acoustic VAD and saved caption clock only."""
    if not 0 <= max_windows <= 12 or not 1000 <= max_total_ms <= 120000:
        raise ValueError("Invalid Qwen shadow decode budget")
    items = []
    for row in final_segments:
        a, b = row.get("start_ms"), row.get("end_ms")
        if (type(a) is not int or type(b) is not int or
                not 0 <= a < b <= duration_ms):
            raise ValueError("Invalid final caption timestamp")
        if row.get("text", "").strip():
            items.append(Utterance(row["text"], a, b))
    windows = uncovered_voice_intervals(
        vad_regions, items, duration_ms,
        min_uncovered_ms=min_ms,
    )
    selected = []
    budget = max_total_ms
    for left, right in windows:
        if len(selected) >= max_windows or budget < min_ms:
            break
        start = max(0, left - 180)
        end = min(duration_ms, right + 180, start + min(budget, 10000))
        if end - start < min_ms:
            continue
        selected.append({
            "start_ms": start, "end_ms": end,
            "reason": "vad_speech_without_confirmed_caption",
        })
        budget -= end - start
    return selected


def silero_vad(samples) -> list[dict]:
    """Run a fresh audio-only VAD, without using AMI word references."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "Shadow gap discovery requires faster-whisper's VAD in the "
            "dedicated experiment environment") from exc
    return get_speech_timestamps(
        samples, VadOptions(min_speech_duration_ms=200,
                            min_silence_duration_ms=450,
                            speech_pad_ms=150),
        sampling_rate=16000,
    )


def shadow(
    audio: Path, audit: Path, *, language: str = "auto",
    max_windows: int = 6, infer: bool = False,
    model_id: str = MODEL, allow_model_download: bool = False,
    vad_provider=None, transcriber=None,
) -> dict:
    audio = audio.resolve(strict=True)
    audit = audit.resolve(strict=True)
    samples = load_wav(audio)
    duration = round(samples.size / 16)
    data = json.loads(audit.read_text(encoding="utf-8"))
    if data.get("schema") != "meeting-live-caption-segment-audit-v1":
        raise ValueError("Expected a private live segment audit")
    actual_audio = data.get("source_audio")
    if actual_audio and Path(actual_audio).name != audio.name:
        raise ValueError("Caption audit belongs to a different audio source")
    if abs(float(data.get("source_audio_duration_s", 0)) * 1000 - duration) > 150:
        raise ValueError("Caption audit and source duration are inconsistent")
    final = data.get("final_segments")
    if not isinstance(final, list):
        raise ValueError("Caption audit is missing final segments")
    vad = (vad_provider or silero_vad)(samples)
    windows = select_gaps(vad, final, duration, max_windows=max_windows)
    candidates = []
    if infer and windows:
        if transcriber is None:
            if not Path(model_id).is_dir() and (
                not allow_model_download or model_id != MODEL
            ):
                raise RuntimeError("Qwen weights require a local folder or explicit download")
            import torch
            from qwen_asr import Qwen3ASRModel
            transcriber = Qwen3ASRModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                max_inference_batch_size=1,
                max_new_tokens=512,
            )
        for item in windows:
            left, right = item["start_ms"], item["end_ms"]
            clipped = samples[left * 16:right * 16].copy()
            tick = time.perf_counter()
            raw = transcriber.transcribe(
                audio=(clipped, 16000),
                language=qwen_language_name(language),
            )
            text = str(getattr(raw[0], "text", "") or "").strip() if raw else ""
            candidates.append({
                **item,
                "candidate_text": text,
                "candidate_language": str(getattr(raw[0], "language", "") or "")
                    if raw else "",
                "wall_seconds": round(time.perf_counter() - tick, 3),
                "human_review_required": True,
            })
    return {
        "schema": SCHEMA,
        "source_audio_sha256": checksum(audio),
        "segment_audit_sha256": checksum(audit),
        "model": model_id if infer else None,
        "infer_enabled": infer,
        "source_is_model_independent_vad": True,
        "reference_words_used_to_select_windows": False,
        "detected_candidate_windows": len(windows),
        "candidate_windows": windows,
        "candidate_transcripts": candidates,
        "auto_replaced_live_text": False,
        "limitations": (
            "VAD activity does not establish intelligible speech or missing "
            "words. Only human listening and independently verified, "
            "time-local text reference can confirm an improvement."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--infer", action="store_true")
    options = parser.parse_args(argv)
    root = Path("benchmark/fixed195").resolve()
    audio, audit, output = (options.audio.resolve(),
                            options.segment_audit.resolve(),
                            options.output.resolve())
    if (any(not p.is_relative_to(root) for p in (audio, audit, output))
            or output.suffix != ".json" or output.exists()):
        parser.error("Private E-drive benchmark input and NEW private JSON output required")
    result = shadow(
        audio, audit, language=options.language,
        max_windows=options.max_windows,
        model_id=options.model_id,
        infer=options.infer,
        allow_model_download=options.allow_model_download,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as dest:
        json.dump(result, dest, ensure_ascii=False, indent=2)
        dest.write("\n")
    print(json.dumps({
        "candidate_windows": result["detected_candidate_windows"],
        "infer_enabled": result["infer_enabled"],
        "auto_replaced_live_text": False,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

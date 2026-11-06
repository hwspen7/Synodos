"""Compare recognition error, glossary recall, language labels and latency.

Offline scoring never loads ASR models. Inference requires --infer explicitly.
Manifest JSONL: {"id":"sample-1", "audio":"clip.wav", "reference":"...",
                 "language":"en", "terms":["product"],
                 "hypotheses":{"whisper":{"text":"...", "language":"en"}}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase07.quality import score_pair, aggregate


def load_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2:
            raise ValueError("Benchmark audio must be 16 kHz mono PCM16 WAV")
        data = stream.readframes(stream.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def inference_engine(name: str, args):
    if name == "whisper":
        from speech.whisper import WhisperEngine
        return WhisperEngine(args.whisper_model, args.device, args.compute_type)
    if name == "qwen3":
        from speech.qwen import QwenEngine
        # The configured Qwen ASR adapter is an optional dependency.
        return QwenEngine(args.qwen_model, args.qwen_device, aligner=args.qwen_aligner)
    raise ValueError(f"Unsupported ASR engine {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", choices=["whisper", "qwen3"], default=["whisper", "qwen3"])
    parser.add_argument("--infer", action="store_true", help="Load models and run real audio inference")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--qwen-model", default="Qwen3-ASR-0.6B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-whisper", action="store_true",
                        help="Load the Whisper model before timed samples; report cold-start separately")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--qwen-aligner", action="store_true")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.manifest.resolve():
        parser.error("Output must not overwrite the manifest")
    if not args.manifest.is_file():
        parser.error("Manifest not found")
    manifest = []
    for lineno, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSONL line {lineno}: {exc.msg}")
        if not isinstance(item, dict) or not isinstance(item.get("reference"), str):
            parser.error(f"Manifest line {lineno} is missing reference text")
        manifest.append(item)
    if not manifest:
        parser.error("Empty benchmark manifest")
    if len(manifest) > 10000:
        parser.error("Benchmark limited to 10000 samples")

    models = {name: inference_engine(name, args) for name in args.engines} if args.infer else {}
    cold_start_ms = None
    if args.infer and args.warmup_whisper and "whisper" in models:
        started = time.perf_counter()
        # Explicitly load model weights before measuring per-case inference time.
        # First CUDA kernel / VAD initialization may still affect the first sample.
        models["whisper"]._get_model()
        cold_start_ms = round((time.perf_counter() - started) * 1000, 2)
        print(f"Whisper model preloaded in {cold_start_ms:.0f} ms", flush=True)
    results: dict[str, list[dict]] = {name: [] for name in args.engines}
    for index, item in enumerate(manifest):
        for name in args.engines:
            try:
                if args.infer:
                    if not isinstance(item.get("audio"), str):
                        raise ValueError("Audio path missing for inference")
                    wav = (args.manifest.parent / item["audio"]).resolve()
                    if not wav.is_relative_to(args.manifest.parent.resolve()):
                        raise ValueError("Audio path must stay under the manifest directory")
                    samples = load_wav(wav)
                    started = time.perf_counter()
                    prediction = models[name].recognize(samples, language=item.get("language_mode", "auto"),
                                                         terms="\n".join(item.get("terms", [])))
                    latency_ms = (time.perf_counter() - started) * 1000
                    hypothesis = " ".join(piece.text for piece in prediction.utterances)
                    predicted_language = prediction.language
                else:
                    prediction = item.get("hypotheses", {}).get(name)
                    if isinstance(prediction, str):
                        prediction = {"text": prediction}
                    if not isinstance(prediction, dict) or not isinstance(prediction.get("text"), str):
                        raise ValueError("Offline hypothesis missing")
                    hypothesis = prediction["text"]
                    predicted_language = prediction.get("language", "")
                    latency_ms = prediction.get("latency_ms")
                score = score_pair(item["reference"], hypothesis, terms=item.get("terms", []),
                                   language=item.get("language", ""))
                score.update({"index": index, "expected_language": item.get("language", ""),
                              "predicted_language": predicted_language, "latency_ms": latency_ms})
                results[name].append(score)
            except Exception as exc:
                # No raw transcript or audio path is written into the report.
                results[name].append({"index": index, "error": type(exc).__name__})
                print(f"Sample {index} / {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    report = {"schema": "meeting-phase07-benchmark-v1", "infer": args.infer,
              "token_metric_policy": "v2: Thai token error unavailable without a word segmenter",
              "whisper_model_load_ms": cold_start_ms,
              "samples": len(manifest), "engines": {
                  name: {"summary": aggregate(rows), "samples": rows}
                  for name, rows in results.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: data["summary"] for name, data in report["engines"].items()}, indent=2))
    return 1 if any(data["summary"]["failed"] for data in report["engines"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

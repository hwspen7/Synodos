"""Diagnose a real Faster-Whisper inference on the current local Python environment.

This test does not use the microphone, Ollama, PostgreSQL, or any web service.
It loads the configured ASR model and forces model decoding, because simply
constructing WhisperModel may not load the required CUDA/cuDNN libraries.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import platform
import site
import sys
import traceback

import numpy as np


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def dll_locations() -> None:
    if os.name != "nt":
        return
    packages = [Path(value) for value in site.getsitepackages()]
    targets = [
        ("cuBLAS", "cublas64_12.dll"),
        ("cuBLAS Lt", "cublasLt64_12.dll"),
        ("cuDNN", "cudnn_ops64_9.dll"),
    ]
    directories = []
    for package in packages:
        directories.extend([
            package / "torch" / "lib",
            package / "ctranslate2",
            package / "nvidia" / "cublas" / "bin",
            package / "nvidia" / "cudnn" / "bin",
        ])
    for label, filename in targets:
        matches = [str(directory / filename) for directory in directories if (directory / filename).is_file()]
        print(f"{label} ({filename}):", ", ".join(matches) if matches else "not found in common Python package folders")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default=None)
    args = parser.parse_args()
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print("Python:", sys.executable, flush=True)
    print("OS:", platform.platform(), flush=True)
    for package in ("faster-whisper", "ctranslate2", "torch"):
        print(f"{package}: {package_version(package)}", flush=True)
    dll_locations()

    try:
        import ctranslate2
        print("CTranslate2 CUDA devices:", ctranslate2.get_cuda_device_count(), flush=True)
        from faster_whisper import WhisperModel
        print(f"Loading Whisper {args.model} on {args.device} ({compute_type})...", flush=True)
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
        print("Running an actual inference pass (silence; no microphone required)...", flush=True)
        samples = np.zeros(3 * 16000, dtype=np.float32)
        segments, _ = model.transcribe(
            samples, language="en", vad_filter=False,
            beam_size=1, word_timestamps=False,
            condition_on_previous_text=False,
        )
        # Faster-Whisper performs part of inference lazily; materialize results.
        list(segments)
        print("PASS: Faster-Whisper inference completed on", args.device, flush=True)
        return 0
    except Exception:
        print("FAIL: Faster-Whisper inference did not complete.", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

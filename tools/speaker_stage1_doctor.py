"""Check optional speaker comparison prerequisites without downloading weights.

This is a dependency diagnostic, not a model accuracy or GPU benchmark.
"""
from __future__ import annotations

import argparse
from importlib import metadata, util
import json
import sys


MODULES = {
    "camplus": ("numpy", "torch", "funasr"),
    "ecapa": ("numpy", "torch", "torchaudio", "speechbrain"),
    "nemotron3": ("numpy", "torch", "transformers", "accelerate"),
}


def check(model: str, *, device: str = "cpu") -> dict:
    if model not in MODULES or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported speaker doctor model or device")
    missing = []
    installed = {}
    for module in MODULES[model]:
        if util.find_spec(module) is None:
            missing.append(module)
            continue
        try:
            installed[module] = metadata.version(module)
        except metadata.PackageNotFoundError:
            installed[module] = "installed_version_unknown"
    cuda_available = None
    if "torch" not in missing:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, OSError, RuntimeError):
            cuda_available = False
    compatible_transformers = None
    if model == "nemotron3" and "transformers" not in missing:
        try:
            from packaging.version import Version
            compatible_transformers = Version(installed["transformers"]) >= Version("5.18.0")
        except (ImportError, ValueError, KeyError):
            compatible_transformers = False
    ok = (
        sys.version_info >= (3, 12)
        and not missing
        and (device != "cuda" or cuda_available is True)
        and compatible_transformers is not False
    )
    return {
        "model": model,
        "device": device,
        "python": sys.version.split()[0],
        "installed": installed,
        "missing": missing,
        "cuda_available": cuda_available,
        "compatible_transformers": compatible_transformers,
        "dependencies_ready": ok,
        "model_weights_tested": False,
        "real_audio_inference_tested": False,
        "note": (
            "This check never downloads weights, executes a model, or "
            "proves speaker accuracy. A reference-backed comparison is required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODULES), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = check(args.model, device=args.device)
    print(json.dumps(report, indent=2))
    return 0 if report["dependencies_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

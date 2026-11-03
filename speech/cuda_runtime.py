"""Register trusted local CUDA DLL directories on Windows before GPU inference.

PyTorch's Windows wheel often bundles CUDA 12 cuBLAS and cuDNN 9 inside
torch/lib. CTranslate2 loads these libraries independently from PyTorch,
so model construction alone does not confirm that they are discoverable.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

_DLL_DIRECTORY_HANDLES: list[object] = []
_CONFIGURED = False


def candidate_directories(prefix: str | Path, cuda_path: str = "") -> tuple[Path, ...]:
    root = Path(prefix)
    installed = root / "Lib" / "site-packages"
    candidates = [
        installed / "torch" / "lib",
        installed / "nvidia" / "cublas" / "bin",
        installed / "nvidia" / "cudnn" / "bin",
    ]
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    return tuple(candidates)


def prepare_windows_cuda_runtime() -> tuple[str, ...]:
    """Add local wheel and CUDA Toolkit DLL folders to the active loader path."""
    global _CONFIGURED
    if os.name != "nt" or _CONFIGURED:
        return ()
    _CONFIGURED = True
    discovered = []
    previous_path = os.environ.get("PATH", "")
    for folder in candidate_directories(sys.prefix, os.environ.get("CUDA_PATH", "")):
        if not folder.is_dir():
            continue
        if not any((folder / filename).exists() for filename in (
            "cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops64_9.dll"
        )):
            continue
        try:
            handle = os.add_dll_directory(str(folder))
        except (AttributeError, OSError):
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        discovered.append(str(folder))
    if discovered:
        # CTranslate2 performs its own DLL loading, so retain a PATH entry
        # in addition to the Python add_dll_directory handles.
        os.environ["PATH"] = os.pathsep.join((*discovered, previous_path))
    return tuple(discovered)

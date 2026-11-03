"""Optional, offline second-opinion diarization on an existing WAV recording.

This is a separate acoustic model from the live FunASR/CAM++ tracker.
It is manually triggered after recording, does not rewrite the transcript,
and does not assume that a resumed meeting's WAV starts at transcript t=0.

Requires an opt-in pyannote Community-1 installation and acceptance of the
Hugging Face model terms. No implicit package installs or hidden uploads.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any

_FILENAME = re.compile(r"^[0-9a-f]{32}\.wav$")
_MODEL_ID = "pyannote/speaker-diarization-community-1"
_PIPELINE: Any = None
_MODEL_LOCK = Lock()
_RUN_LOCK = Lock()


class OfflineDiarizationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker: str


def validate_recording(root: Path, meeting_id: str, filename: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", meeting_id):
        raise ValueError("Invalid meeting identifier")
    if not _FILENAME.fullmatch(filename):
        raise ValueError("Invalid WAV recording name")
    root = Path(root).resolve()
    path = (root / meeting_id / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("Saved WAV recording was not found")
    if path.stat().st_size > 1_000_000_000:
        raise ValueError("WAV exceeds the offline processing limit")
    return path


def _pipeline():
    global _PIPELINE
    with _MODEL_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise OfflineDiarizationUnavailable(
                "Speaker refinement requires HF_TOKEN and acceptance of the "
                "pyannote Community-1 model terms on Hugging Face."
            )
        try:
            from pyannote.audio import Pipeline
        except (ImportError, OSError) as exc:
            raise OfflineDiarizationUnavailable(
                "Optional pyannote.audio is not installed. Install it in a "
                "separate validated environment before enabling speaker refinement."
            ) from exc
        try:
            result = Pipeline.from_pretrained(_MODEL_ID, token=token)
            if result is None:
                raise RuntimeError("Model access was rejected")
        except Exception as exc:
            raise OfflineDiarizationUnavailable(
                "Unable to load Community-1. Check model access approval, "
                "HF_TOKEN, network access on first download and local model cache."
            ) from exc
        # CPU is deliberately the default. The local GPU is already shared
        # by Whisper and Saluki; GPU use must be explicitly opted in.
        if os.getenv("MEETING_OFFLINE_DIARIZATION_DEVICE", "cpu").lower() == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                result.to(torch.device("cuda"))
            except Exception as exc:
                raise OfflineDiarizationUnavailable(
                    "Community-1 CUDA loading failed; choose CPU or check its dependencies"
                ) from exc
        _PIPELINE = result
        return result


def normalize_turns(diarization: Any, *, max_turns: int = 10000) -> list[dict]:
    """Normalize version-dependent pyannote outputs without forcing identities."""
    if hasattr(diarization, "itertracks"):
        source = ((segment, speaker)
                  for segment, _, speaker in diarization.itertracks(yield_label=True))
    elif hasattr(diarization, "__iter__"):
        source = iter(diarization)
    else:
        raise ValueError("Unsupported diarization output")
    turns: list[SpeakerTurn] = []
    for item in source:
        try:
            span, speaker = item
            start_ms = round(float(span.start) * 1000)
            end_ms = round(float(span.end) * 1000)
            speaker = str(speaker)
        except (TypeError, AttributeError, ValueError, OverflowError) as exc:
            raise ValueError("Malformed speaker time span") from exc
        if not speaker or start_ms < 0 or end_ms <= start_ms:
            continue
        turns.append(SpeakerTurn(start_ms, end_ms, speaker))
        if len(turns) > max_turns:
            raise ValueError("Speaker timeline exceeds safety limit")
    turns.sort(key=lambda t: (t.start_ms, t.end_ms, t.speaker))
    return [{"start_ms": t.start_ms, "end_ms": t.end_ms, "speaker": t.speaker}
            for t in turns]


def run_offline_refinement(
    root: Path, meeting_id: str, filename: str,
    *,
    backend: Any = None,
) -> dict:
    path = validate_recording(root, meeting_id, filename)
    model = backend if backend is not None else _pipeline()
    with _RUN_LOCK:
        output = model(str(path))
    annotations = getattr(output, "exclusive_speaker_diarization", None)
    if annotations is None:
        annotations = getattr(output, "speaker_diarization", output)
    turns = normalize_turns(annotations)
    return {
        "model": _MODEL_ID if backend is None else "test_or_custom_diarizer",
        "source_filename": filename,
        "relative_to_recording": True,
        "speaker_turns": turns,
        "speaker_count": len({turn["speaker"] for turn in turns}),
        "review_required": True,
        "auto_applied": False,
        "notice": (
            "Speaker labels are local to this WAV and are NOT participant identities. "
            "A resumed meeting may have a different transcript origin. "
            "Verify against audio before assigning any saved speaker identity."
        ),
    }

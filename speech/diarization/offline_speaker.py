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


def find_overlapping_speaker_intervals(turns: list[dict]) -> list[dict]:
    """Return intervals with >=2 different active speaker IDs, including 3+.

    Same-speaker overlapping tracks count only once. Anonymous speaker
    labels are used only to detect concurrency, not real-world identity.
    """
    events: dict[int, list[tuple[str, int]]] = {}
    for turn in turns:
        a, b, speaker = (
            turn.get("start_ms"), turn.get("end_ms"), turn.get("speaker"),
        )
        if (type(a) is not int or type(b) is not int or not 0 <= a < b
                or not isinstance(speaker, str) or not speaker):
            raise ValueError("Invalid offline speaker time range")
        events.setdefault(a, []).append((speaker, 1))
        events.setdefault(b, []).append((speaker, -1))
    active: dict[str, int] = {}
    result = []
    previous = None
    for t in sorted(events):
        active_count = sum(value > 0 for value in active.values())
        if previous is not None and t > previous and active_count >= 2:
            if (result and result[-1]["end_ms"] == previous
                    and result[-1]["active_speaker_count"] == active_count):
                result[-1]["end_ms"] = t
            else:
                result.append({
                    "start_ms": previous, "end_ms": t,
                    "active_speaker_count": active_count,
                })
        for speaker, delta in events[t]:
            new_count = active.get(speaker, 0) + delta
            if new_count <= 0:
                active.pop(speaker, None)
            else:
                active[speaker] = new_count
        previous = t
    return result


def run_offline_refinement(
    root: Path, meeting_id: str, filename: str,
    *,
    backend: Any = None,
) -> dict:
    path = validate_recording(root, meeting_id, filename)
    if backend is None:
        # Hard dependency isolation. pyannote/NeMo must not overwrite the
        # already-validated Whisper + CUDA environment.
        import json
        import subprocess
        python = os.getenv("MEETING_SPEAKER_REFINER_PYTHON", "").strip()
        if not python:
            raise OfflineDiarizationUnavailable(
                "Configure MEETING_SPEAKER_REFINER_PYTHON to point to a dedicated "
                "pyannote Python environment."
            )
        executable = Path(python).expanduser()
        if not executable.is_file():
            raise OfflineDiarizationUnavailable("Dedicated speaker Python executable was not found")
        try:
            process = subprocess.run(
                [str(executable), "-m", "speech.diarization.offline_speaker_cli", str(path)],
                cwd=str(Path(__file__).resolve().parents[2]),
                capture_output=True, text=True, timeout=1800, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OfflineDiarizationUnavailable(
                "Dedicated speaker refinement process failed to start or timed out"
            ) from exc
        if process.returncode:
            try:
                detail = json.loads(process.stdout).get("error", "")
            except (ValueError, TypeError, AttributeError):
                detail = ""
            raise OfflineDiarizationUnavailable(
                "Dedicated speaker model failed: " + str(detail)[:240]
            )
        try:
            payload = json.loads(process.stdout)
            raw = payload["speaker_turns"]
            if not isinstance(raw, list) or len(raw) > 10000:
                raise ValueError("Invalid speaker output")
            checked = []
            for item in raw:
                if not isinstance(item, dict):
                    raise ValueError("Invalid speaker output")
                a, b, speaker = item["start_ms"], item["end_ms"], item["speaker"]
                if (type(a) is not int or type(b) is not int or
                        a < 0 or b <= a or b > 1000 * 60 * 60 * 24 or
                        not isinstance(speaker, str) or not speaker or len(speaker) > 90):
                    raise ValueError("Invalid speaker interval")
                checked.append({"start_ms": a, "end_ms": b, "speaker": speaker})
            turns = sorted(checked, key=lambda item: item["start_ms"])
            exclusive = payload.get("exclusive_speaker_turns")
            if exclusive is not None:
                if not isinstance(exclusive, list) or len(exclusive) > 10000:
                    raise ValueError("Invalid exclusive speaker output")
                exclusive_checked = []
                for item in exclusive:
                    a, b, speaker = item["start_ms"], item["end_ms"], item["speaker"]
                    if (type(a) is not int or type(b) is not int
                            or not 0 <= a < b <= 1000 * 60 * 60 * 24
                            or not isinstance(speaker, str) or not speaker
                            or len(speaker) > 90):
                        raise ValueError("Invalid exclusive speaker interval")
                    exclusive_checked.append({
                        "start_ms": a, "end_ms": b, "speaker": speaker,
                    })
                exclusive = sorted(
                    exclusive_checked, key=lambda item: item["start_ms"],
                )
        except (ValueError, TypeError, KeyError) as exc:
            raise OfflineDiarizationUnavailable(
                "Dedicated speaker process returned invalid data"
            ) from exc
    else:
        with _RUN_LOCK:
            output = backend(str(path))
        annotations = getattr(output, "speaker_diarization", None)
        if annotations is None:
            annotations = getattr(output, "exclusive_speaker_diarization", output)
        turns = normalize_turns(annotations)
        exclusive_annotation = getattr(
            output, "exclusive_speaker_diarization", None,
        )
        exclusive = (
            normalize_turns(exclusive_annotation)
            if exclusive_annotation is not None else None
        )
    return {
        "model": _MODEL_ID if backend is None else "test_or_custom_diarizer",
        "source_filename": filename,
        "relative_to_recording": True,
        "speaker_turns": turns,
        "exclusive_speaker_turns": exclusive,
        "overlap_intervals": find_overlapping_speaker_intervals(turns),
        "overlap_turn_count": len(find_overlapping_speaker_intervals(turns)),
        "speaker_count": len({turn["speaker"] for turn in turns}),
        "review_required": True,
        "auto_applied": False,
        "notice": (
            "Speaker labels are local to this WAV and are NOT participant identities. "
            "A resumed meeting may have a different transcript origin. "
            "Verify against audio before assigning any saved speaker identity."
        ),
    }

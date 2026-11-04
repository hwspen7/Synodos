"""Nonblocking, opt-in ECAPA shadow for live CAM++ speaker labels.

The live processor NEVER waits for ECAPA. One globally admitted meeting
may run the candidate worker at a time (16 GB GPU budget). Failures, queue
pressure and missing optional dependencies cannot change the transcript.
"""
from __future__ import annotations

from collections import Counter
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np

from speech.ecapa_shadow_worker import PREFIX

_ADMISSION = threading.BoundedSemaphore(1)
_VALID_OUTCOMES = frozenset({
    "unlabeled", "no_embedding", "capacity", "seeded", "no_comparator",
    "rank_conflict", "rank_match", "low_energy", "error",
})
_MAX_QUEUE = 6
_MAX_SECONDS = 6
# These are minimum electrical levels, not trained speech/VAD thresholds.
# The ASR-confirmed sentence is the primary speech evidence.
_MIN_RMS = 0.001
_FRAME_SAMPLES = 3200
_SELECTION_SECONDS = 3


def select_shadow_window(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
    """Choose the most active 2-3 second excerpt of an ASR-confirmed turn.

    Short-term energy identifies where a word-level segment is loudest.
    Never fabricate speech from a quiet excerpt, and never normalize
    noise blindly to force a successful voiceprint.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if (sample_rate != 16000 or not 2 * sample_rate <= samples.size
            or samples.size > 120 * sample_rate
            or not np.isfinite(samples).all()
            or np.max(np.abs(samples)) > 1.001):
        return None
    size = min(_SELECTION_SECONDS * sample_rate, samples.size)
    step = sample_rate // 2
    best = None
    for start in range(0, samples.size - size + 1, step):
        window = samples[start:start + size]
        rms = float(np.sqrt(np.mean(window * window)))
        if rms < _MIN_RMS:
            continue
        frame_count = window.size // _FRAME_SAMPLES
        if not frame_count:
            continue
        frames = window[:frame_count * _FRAME_SAMPLES].reshape(
            frame_count, _FRAME_SAMPLES
        )
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        active_fraction = float(np.mean(
            frame_rms >= max(_MIN_RMS, 0.20 * float(frame_rms.max()))
        ))
        if active_fraction < 0.40:
            continue
        # Prefer broad activity over one short loud transient.
        score = (active_fraction, min(rms, 0.2))
        if best is None or score > best[0]:
            best = (score, start)
    if best is None:
        return None
    start = best[1]
    return np.array(samples[start:start + size], dtype=np.float32, copy=True)


def _enabled() -> bool:
    return os.getenv("MEETING_ECAPA_SHADOW", "0").strip().lower() in (
        "1", "true", "yes",
    )


def _worker_settings() -> tuple[str, str, Path, dict]:
    repo = Path(__file__).resolve().parents[1]
    device = os.getenv("MEETING_ECAPA_SHADOW_DEVICE", "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        raise ValueError("MEETING_ECAPA_SHADOW_DEVICE must be cpu or cuda")
    default_python = repo / ".venv-speaker-ecapa" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    python = Path(os.getenv("MEETING_ECAPA_SHADOW_PYTHON", str(default_python)))
    if not python.is_file():
        raise FileNotFoundError("Dedicated ECAPA Python environment is missing")
    # Default all model caches to the same disk as the project checkout.
    # Never inherit a user's previous C: Hugging Face cache by accident.
    default_cache = Path(repo.anchor) / "Meeting-AI-Cache"
    cache = Path(os.getenv("MEETING_ECAPA_SHADOW_CACHE_ROOT", str(default_cache))).resolve()
    if os.name == "nt" and cache.drive.lower() != repo.drive.lower():
        raise ValueError("ECAPA shadow cache must be on the project drive")
    mapping = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_XET_CACHE": cache / "huggingface" / "xet",
        "TORCH_HOME": cache / "torch",
        "MODELSCOPE_CACHE": cache / "modelscope",
        "XDG_CACHE_HOME": cache / "xdg",
        "TEMP": cache / "temp",
        "TMP": cache / "temp",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({key: str(value) for key, value in mapping.items()})
    return str(python), device, repo, env


class EcapaShadowSession:
    """One-session, one-process, in-memory-only asynchronous ECAPA diagnostic."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._jobs: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=_MAX_QUEUE)
        self._counts: Counter[str] = Counter()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._available = True
        self._observed_labels: set[int] = set()

    def submit(self, label: int, audio: np.ndarray) -> bool:
        """Never block or mutate main speaker IDs, text or embeddings."""
        if type(label) is not int or label <= 0 or self._stopped.is_set():
            return False
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if (not 32000 <= samples.size <= 16000 * _MAX_SECONDS
                or not np.isfinite(samples).all()
                or np.max(np.abs(samples)) > 1.001):
            with self._lock:
                self._counts["invalid_audio"] += 1
            return False
        # Fast energy gate reduces background GPU/CPU work but is not a VAD.
        # Audio is already a selected 2-3s ASR-confirmed voice excerpt.
        # The old 0.008 RMS gate rejected quiet but legitimate AMI speech.
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms < _MIN_RMS:
            with self._lock:
                self._counts["low_energy"] += 1
            return False
        pcm = (np.clip(samples, -1.0, 0.999969) * 32768.0).astype("<i2").tobytes()
        with self._lock:
            if not self._available or self._stopped.is_set():
                return False
            try:
                self._jobs.put_nowait((label, pcm))
            except queue.Full:
                self._counts["queue_full"] += 1
                return False
            self._counts["submitted"] += 1
            self._observed_labels.add(label)
            if rms < 0.008:
                self._counts["accepted_quiet_speech"] += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, daemon=True,
                    name="meeting-ecapa-shadow",
                )
                self._thread.start()
            return True

    def _start_process(self) -> subprocess.Popen:
        python, device, repo, env = _worker_settings()
        return subprocess.Popen(
            [python, "-u", "-m", "speech.ecapa_shadow_worker", device],
            cwd=str(repo), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )

    def _exchange(self, label: int, pcm: bytes) -> str:
        proc = self._process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Shadow worker unavailable")
        request = {
            "op": "observe",
            "sample_rate": 16000,
            "label": label,
            "pcm16_b64": base64.b64encode(pcm).decode("ascii"),
        }
        proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        proc.stdin.flush()
        # Dependency/model libraries may print startup messages to stdout.
        # Accept only our framed protocol; do not log arbitrary output.
        for _ in range(24):
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("Shadow worker terminated")
            if line.startswith(PREFIX.encode("utf-8")):
                result = json.loads(line[len(PREFIX):])
                outcome = result.get("outcome")
                if outcome not in _VALID_OUTCOMES:
                    raise ValueError("Unrecognized shadow response")
                return outcome
        raise RuntimeError("Shadow protocol out of sync")

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                label, pcm = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stopped.is_set():
                break
            try:
                if self._process is None:
                    child = self._start_process()
                    with self._lock:
                        if self._stopped.is_set():
                            child.terminate()
                            break
                        self._process = child
                outcome = self._exchange(label, pcm)
                with self._lock:
                    self._counts["completed"] += 1
                    self._counts[outcome] += 1
            except Exception:
                with self._lock:
                    self._counts["errors"] += 1
                    self._available = False
                break
            finally:
                self._jobs.task_done()

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": "shadow_only_no_identity_changes",
                "submitted": self._counts["submitted"],
                "completed": self._counts["completed"],
                "observed_camplus_speaker_count": len(self._observed_labels),
                "accepted_quiet_speech": self._counts["accepted_quiet_speech"],
                "rank_match": self._counts["rank_match"],
                "rank_conflict": self._counts["rank_conflict"],
                "seeded": self._counts["seeded"],
                "no_comparator": self._counts["no_comparator"],
                "skipped_queue_full": self._counts["queue_full"],
                "skipped_invalid_audio": self._counts["invalid_audio"],
                "skipped_low_energy": self._counts["low_energy"],
                "errors": self._counts["errors"],
                "worker_available": self._available,
                "remaining": self._jobs.qsize(),
                "model_agreement_is_not_accuracy": True,
            }

    def close(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        with self._lock:
            proc = self._process
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        # The daemon thread and subprocess must never delay meeting shutdown.
        # Clear queued audio immediately; it is never persisted.
        while True:
            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                break
        _ADMISSION.release()


def create_shadow_session() -> EcapaShadowSession | None:
    if not _enabled():
        return None
    if not _ADMISSION.acquire(blocking=False):
        return None
    return EcapaShadowSession()

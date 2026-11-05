"""Private persistent local Qwen subprocess bridge for the Meeting main venv.

Qwen's Python and CUDA packages are isolated on E:. Requests use guarded
stdin/stdout messages; worker output cannot become a meeting network API.
"""
from __future__ import annotations

import atexit
import base64
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading

import numpy as np
from speech.contracts import Recognition, Utterance, Word

PREFIX = "MEETING_QWEN_WORKER_V1:"
PREFIX_BYTES = PREFIX.encode("ascii")
MAX_SAMPLES = 16000 * 120
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def parse_worker_stdout(raw: bytes) -> dict | None:
    """Decode only tagged ASCII protocol frames, never native log bytes.

    Some Windows CUDA/native libraries write non-UTF8 or code-page bytes
    directly to stdout. They are untrusted diagnostic noise, not frames.
    Native output without a newline may precede the worker's frame.
    """
    if not isinstance(raw, bytes):
        raise TypeError("Qwen stdout must be read as binary")
    marker = raw.find(PREFIX_BYTES)
    if marker < 0:
        return None
    encoded = raw[marker + len(PREFIX_BYTES):].strip()
    if not encoded or len(encoded) > MAX_RESPONSE_BYTES:
        return {"id": -1, "ok": False, "error_type": "InvalidFrameSize"}
    try:
        response = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"id": -1, "ok": False, "error_type": "InvalidFrameEncoding"}
    if not isinstance(response, dict) or type(response.get("id")) is not int:
        return {"id": -1, "ok": False, "error_type": "InvalidFrameStructure"}
    return response


def decode_recognition(payload):
    if not isinstance(payload, dict):
        raise ValueError("Invalid Qwen response")
    utterances = []
    for row in payload["utterances"]:
        words = tuple(Word(
            str(w["text"]), int(w["start_ms"]), int(w["end_ms"])
        ) for w in row.get("words", ()))
        utterances.append(Utterance(
            str(row["text"]), int(row["start_ms"]), int(row["end_ms"]),
            str(row.get("language", "und")), words,
            str(row.get("timestamp_quality", "segment")),
            row.get("confidence"), row.get("language_confidence"),
        ))
    return Recognition(
        tuple(utterances), str(payload.get("language", "und")),
        payload.get("confidence"), "qwen3",
        payload.get("speech_detected"),
        tuple((int(a), int(b)) for a, b in payload.get("speech_intervals_ms", ())),
    )


class QwenWorkerClient:
    def __init__(self, python_path: str, model: str = "Qwen/Qwen3-ASR-1.7B"):
        python = Path(python_path).resolve()
        if not python.is_file():
            raise ValueError("Independent Qwen Python executable not found")
        if os.name == "nt" and python.drive.lower() != "e:":
            raise ValueError("Qwen Python must be on E drive")
        if model != "Qwen/Qwen3-ASR-1.7B":
            raise ValueError("Only Qwen3-ASR-1.7B is approved for dual ASR")
        self.python = str(python)
        self.lock = threading.RLock()
        self.proc = None
        self.log = None
        self.queue = Queue()
        self.sequence = 0
        try:
            self.timeout = float(os.getenv("MEETING_HYBRID_QWEN_TIMEOUT_S", "180"))
        except ValueError as exc:
            raise ValueError("Invalid hybrid worker timeout") from exc
        if not math.isfinite(self.timeout) or not 10 <= self.timeout <= 600:
            raise ValueError("Hybrid worker timeout must be 10-600 seconds")
        atexit.register(self.close)

    def _read_output(self, stream, channel):
        """Drain raw bytes and signal failure on EOF without thread crashes."""
        try:
            for raw in stream:
                result = parse_worker_stdout(raw)
                if result is not None:
                    channel.put(result)
        except (OSError, ValueError) as exc:
            channel.put({
                "id": -1, "ok": False,
                "error_type": type(exc).__name__,
            })
        finally:
            # Capture this worker's own queue. A dead worker must never
            # poison a replacement worker's new response queue.
            channel.put(None)

    def _stop(self):
        process = self.proc
        self.proc = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self.log is not None:
            self.log.close()
            self.log = None

    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self._stop()
        path = Path(os.getenv(
            "MEETING_HYBRID_QWEN_LOG_DIR",
            r"E:\AI-Cache\MeetingLogs" if os.name == "nt" else "/tmp"
        )).resolve()
        if os.name == "nt" and path.drive.lower() != "e:":
            raise ValueError("Qwen worker logs must remain on E drive")
        path.mkdir(parents=True, exist_ok=True)
        self.log = (path / "qwen_asr_worker.log").open("a", encoding="utf-8")
        self.queue = Queue()
        worker_env = os.environ.copy()
        worker_env["PYTHONUTF8"] = "1"
        worker_env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            [self.python, "-u", "-m", "speech.qwen_worker"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, bufsize=0, env=worker_env,
        )
        channel = self.queue
        threading.Thread(
            target=self._read_output, args=(self.proc.stdout, channel),
            name="qwen-asr-stdout", daemon=True,
        ).start()

    def close(self):
        with self.lock:
            self._stop()

    def _request(self, command):
        with self.lock:
            self._start()
            self.sequence += 1
            request_id = self.sequence
            try:
                wire = (json.dumps({
                    "id": request_id, **command,
                }, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
                self.proc.stdin.write(wire)
                self.proc.stdin.flush()
                response = self.queue.get(timeout=self.timeout)
                if response is None or response.get("id") != request_id:
                    raise RuntimeError("Qwen worker stopped or response was out of order")
                if not response.get("ok"):
                    raise RuntimeError(
                        "Qwen worker " + str(response.get("error_type", "error"))
                    )
                return response["result"]
            except (OSError, RuntimeError, KeyError, ValueError, Empty) as exc:
                self._stop()
                raise RuntimeError(
                    "Qwen isolated CUDA worker failed or exceeded timeout"
                ) from exc

    def prewarm(self):
        if self._request({"command": "prewarm"}).get("ready") is not True:
            raise RuntimeError("Qwen worker prewarm failed")

    def recognize(self, audio, language="auto", terms=""):
        samples = np.asarray(audio, dtype="<f4")
        if (samples.ndim != 1 or not 0 < samples.size <= MAX_SAMPLES or
                not np.isfinite(samples).all()):
            raise ValueError("Expected bounded mono 16 kHz samples")
        payload = {
            "command": "recognize",
            "audio_f32le": base64.b64encode(samples.tobytes()).decode("ascii"),
            "language": str(language), "terms": str(terms)[:1500],
        }
        return decode_recognition(self._request(payload))

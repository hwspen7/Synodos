"""Independent live Nemotron-3 speaker stream with persistent speaker cache.

Opt-in only. The audio is fed from the accepted WebSocket PCM path, never
from Whisper's confirmed sentences or its reclaimable rolling buffer.

The official Transformers streaming API handles model look-ahead and AOSC
state. Each meeting has its OWN speaker_cache; weights can be shared.
"""
from __future__ import annotations

from functools import lru_cache
import logging
import os
from queue import Empty, Full, Queue
import threading
from typing import Callable

import numpy as np

from speech.diarization.nemotron_activity import validate_activity
from speech.diarization.speaker_backends import normalize_nemotron_turns

logger = logging.getLogger(__name__)
MODEL_ID = "nvidia/Nemotron-3-Diarization"
SAMPLE_RATE = 16000
FRAME_MS = 10
_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _model_resources(mode: str):
    if mode not in ("low_latency", "very_low_latency", "ultra_low_latency"):
        raise ValueError("Unsupported Nemotron streaming mode")
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.set_streaming_mode(mode)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID)
    device = os.getenv("MEETING_NEMOTRON_DEVICE", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("MEETING_NEMOTRON_DEVICE must be cpu or cuda")
    model = model.to(device)
    model.eval()
    if int(processor.feature_extractor.sampling_rate) != SAMPLE_RATE:
        raise ValueError("Nemotron expects 16 kHz PCM")
    return processor, model


class StreamingChunks:
    """Follow NVIDIA's official first/chunk/final input-offset semantics.

    Keeping the trailing look-ahead samples is mandatory: do NOT build
    independent 120 s chunks or reset speaker_cache at window boundaries.
    Inference returns raw logits and a new, per-session cache.
    """

    def __init__(self, processor, infer: Callable):
        self.processor = processor
        self.infer = infer
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_start = 0
        self._received = 0
        self._first = False
        self._closed = False
        self._mel_frame_index = 0
        self._next_start = 0
        self.speaker_cache = None
        self.logits: list[np.ndarray] = []
        self.probabilities: list[np.ndarray] = []

    @property
    def received_samples(self) -> int:
        return self._received

    @property
    def scored_frames(self) -> int:
        return sum(len(chunk) for chunk in self.probabilities)

    def _clip(self, begin: int, end: int) -> np.ndarray:
        a, b = begin - self._pending_start, end - self._pending_start
        if a < 0 or b > len(self._pending) or a > b:
            raise RuntimeError("Nemotron streaming PCM continuity violated")
        return np.ascontiguousarray(self._pending[a:b], dtype=np.float32)

    def _trim(self):
        # Only samples preceding the NEXT inference window can be deleted.
        drop = self._next_start - self._pending_start
        if drop < 0 or drop > len(self._pending):
            raise RuntimeError("Nemotron look-ahead buffer geometry invalid")
        if drop:
            self._pending = self._pending[drop:].copy()
            self._pending_start += drop

    def _call(self, clip: np.ndarray, *, first: bool, last: bool):
        raw, cache = self.infer(clip, first, last, self.speaker_cache)
        logits = np.asarray(raw, dtype=np.float32)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        if (logits.ndim != 2 or logits.shape[1] != 8
                or not np.isfinite(logits).all()):
            raise ValueError("Nemotron returned invalid [T,8] logits")
        self.speaker_cache = cache
        if len(logits):
            self.logits.append(logits.copy())
            # The model returns raw logits, NOT activity probabilities.
            probabilities = 1.0 / (1.0 + np.exp(
                -np.clip(logits, -60.0, 60.0),
            ))
            self.probabilities.append(validate_activity(probabilities))

    def accept(self, pcm: np.ndarray):
        if self._closed:
            raise RuntimeError("Nemotron stream already closed")
        frame = np.asarray(pcm, dtype=np.float32)
        if frame.ndim != 1 or not np.isfinite(frame).all():
            raise ValueError("Require finite mono PCM")
        if not len(frame):
            return
        self._pending = np.concatenate((self._pending, frame))
        self._received += len(frame)
        self._available()

    def _available(self):
        p = self.processor
        if not self._first:
            n = int(p.num_samples_first_audio_chunk)
            if n <= 0:
                raise RuntimeError("Invalid first Nemotron audio chunk size")
            if self._received < n:
                return
            self._call(self._clip(0, n), first=True, last=False)
            self._first = True
            self._mel_frame_index = int(p.num_mel_frames_per_step)
            self._next_start = int(p.audio_chunk_start(self._mel_frame_index))
            self._trim()
        n = int(p.num_samples_per_audio_chunk)
        step = int(p.num_mel_frames_per_step)
        if n <= 0 or step <= 0:
            raise RuntimeError("Invalid subsequent Nemotron chunk geometry")
        while self._next_start + n <= self._received:
            self._call(
                self._clip(self._next_start, self._next_start + n),
                first=False, last=False,
            )
            prior = self._next_start
            self._mel_frame_index += step
            self._next_start = int(p.audio_chunk_start(self._mel_frame_index))
            if self._next_start <= prior:
                raise RuntimeError("Nemotron input position did not advance")
            self._trim()

    def finish(self):
        if self._closed:
            return
        if not self._first:
            if self._received:
                self._call(self._clip(0, self._received), first=True, last=True)
        else:
            # The last chunk must release all remaining audio/look-ahead.
            self._call(
                self._clip(self._next_start, self._received),
                first=False, last=True,
            )
        self._closed = True

    def activity(self) -> np.ndarray:
        if not self.probabilities:
            return np.empty((0, 8), dtype=np.float32)
        return np.concatenate(self.probabilities, axis=0)

    def raw_logits(self) -> np.ndarray:
        if not self.logits:
            return np.empty((0, 8), dtype=np.float32)
        return np.concatenate(self.logits, axis=0)


class NemotronRealtimeWorker:
    """One independent bounded PCM queue and AOSC state per meeting."""

    def __init__(self, *, on_update: Callable[[], None] | None = None,
                 queue_frames: int = 600,
                 runner_factory: Callable | None = None):
        if not 1 <= queue_frames <= 1200:
            raise ValueError("Invalid Nemotron PCM queue budget")
        self._queue: Queue = Queue(maxsize=queue_frames)
        # Count acoustically meaningful PCM blocks, not browser worklet
        # quanta. Small frames are packed into 200ms blocks in FIFO order.
        # This is a bounded staging buffer; it never drops a valid tail.
        self._pcm_block_samples = SAMPLE_RATE // 5
        self._pending_pcm = np.empty(0, dtype=np.float32)
        self._accepted_audio_samples = 0
        self._max_queued_pcm_frames = 0
        self._on_update = on_update
        self._factory = runner_factory
        self._thread = threading.Thread(
            target=self._run, name="nemotron-realtime", daemon=True,
        )
        self._lock = threading.Lock()
        self._state = "created"
        self._error: str | None = None
        self._audio_samples = 0
        self._logits = np.empty((0, 8), dtype=np.float32)
        self._activity = np.empty((0, 8), dtype=np.float32)
        self._turns: list[dict] = []
        self._stop_sent = False
        self._finish_requested = threading.Event()

    def start(self):
        self._thread.start()

    def _queue_pcm_locked(self, samples: np.ndarray) -> bool:
        """Enqueue a PCM block while holding the admission lock."""
        try:
            self._queue.put_nowait(samples.copy())
        except Full:
            self._error = "Nemotron PCM queue full; independent track incomplete"
            self._state = "failed"
            return False
        self._max_queued_pcm_frames = max(
            self._max_queued_pcm_frames, self._queue.qsize(),
        )
        return True

    def feed(self, pcm: np.ndarray) -> bool:
        arr = np.asarray(pcm, dtype=np.float32)
        if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
            raise ValueError("Invalid Nemotron PCM")
        with self._lock:
            if self._state in ("failed", "completed") or self._stop_sent:
                return False
            self._pending_pcm = np.concatenate((self._pending_pcm, arr))
            while len(self._pending_pcm) >= self._pcm_block_samples:
                chunk = self._pending_pcm[:self._pcm_block_samples]
                if not self._queue_pcm_locked(chunk):
                    self._pending_pcm = np.empty(0, dtype=np.float32)
                    return False
                self._pending_pcm = self._pending_pcm[self._pcm_block_samples:].copy()
            self._accepted_audio_samples += len(arr)
            return True

    def _default_runner(self):
        import torch
        mode = os.getenv("MEETING_NEMOTRON_STREAM_MODE", "low_latency")
        processor, model = _model_resources(mode)

        def inference(audio, first, last, cache):
            with _MODEL_LOCK, torch.inference_mode():
                # The shared processor and weights must be serialized
                # across simultaneous meetings. Cache stays per session.
                inputs = processor(
                    audio, sampling_rate=SAMPLE_RATE,
                    is_streaming=True,
                    is_first_audio_chunk=first,
                    is_last_audio_chunk=last,
                )
                out = model(
                    **inputs.to(model.device, dtype=model.dtype),
                    speaker_cache=cache,
                )
            return out.logits.detach().float().cpu().numpy(), out.speaker_cache

        def turns(raw, duration_ms):
            if not len(raw):
                return []
            with _MODEL_LOCK, torch.inference_mode():
                all_logits = torch.from_numpy(raw).unsqueeze(0)
                prediction = processor.extract_speaker_dict(all_logits)[0]
            return normalize_nemotron_turns(
                prediction, duration_ms=duration_ms,
            )

        return StreamingChunks(processor, inference), turns

    def _run(self):
        try:
            with self._lock:
                self._state = "loading"
            stream, extract = (
                self._factory() if self._factory is not None
                else self._default_runner()
            )
            with self._lock:
                self._state = "running"
            while True:
                try:
                    frame = self._queue.get(timeout=0.2)
                except Empty:
                    if self._finish_requested.is_set():
                        break
                    continue
                with self._lock:
                    failed = self._state == "failed"
                if failed:
                    continue
                previous_frames = stream.scored_frames
                stream.accept(frame)
                changed = stream.scored_frames > previous_frames
                with self._lock:
                    self._audio_samples += len(frame)
                    if changed:
                        self._activity = stream.activity()
                if changed and self._on_update is not None:
                    self._on_update()
                if self._finish_requested.is_set() and self._queue.empty():
                    break
            with self._lock:
                failed = self._state == "failed"
            if failed:
                return
            stream.finish()
            activity = stream.activity()
            turns = extract(
                stream.raw_logits(), round(stream.received_samples * 1000 / SAMPLE_RATE),
            )
            with self._lock:
                self._activity = activity
                self._logits = stream.raw_logits()
                self._audio_samples = stream.received_samples
                self._turns = turns
                self._state = "completed"
            if self._on_update is not None:
                self._on_update()
        except Exception as exc:
            logger.exception("Independent Nemotron diarization failed")
            with self._lock:
                self._error = f"{type(exc).__name__}: {str(exc)[:160]}"
                self._state = "failed"

    def request_finish(self) -> None:
        """Enqueue the last partial PCM block before signaling completion.

        An exhausted queue fails closed rather than silently truncating audio.
        """
        with self._lock:
            if not self._stop_sent:
                if len(self._pending_pcm) and self._state not in ("failed", "completed"):
                    self._queue_pcm_locked(self._pending_pcm)
                self._pending_pcm = np.empty(0, dtype=np.float32)
                self._stop_sent = True
        self._finish_requested.set()

    def finish(self, timeout: float = 90.0) -> bool:
        if not self._thread.ident:
            with self._lock:
                self._state = "failed"
                self._error = "Nemotron worker not started"
            return False
        self.request_finish()
        self._thread.join(timeout)
        if self._thread.is_alive():
            with self._lock:
                self._state = "failed"
                self._error = "Nemotron worker finalization exceeded timeout"
        with self._lock:
            return self._state == "completed"

    def is_complete(self) -> bool:
        """Whether final speaker activity covers a successfully closed stream."""
        with self._lock:
            return self._state == "completed"

    def final_turns_snapshot(self) -> list[dict] | None:
        """Return a complete, official speaker-turn result or no result.

        None means inference is still pending/failed. An empty list means a
        successfully completed session with no detected speech.
        """
        with self._lock:
            if self._state != "completed":
                return None
            return [dict(row) for row in self._turns]

    def activity_snapshot(self) -> np.ndarray | None:
        with self._lock:
            if self._state not in ("running", "completed"):
                return None
            return self._activity.copy()

    def report(self) -> dict:
        with self._lock:
            rows = self._turns.copy()
            return {
                "backend": "nemotron3",
                "model_id": MODEL_ID,
                "state": self._state,
                "error": self._error,
                "processed_audio_samples": self._audio_samples,
                "input_audio_samples": self._accepted_audio_samples,
                "max_queued_pcm_frames": self._max_queued_pcm_frames,
                "pending_pcm_samples": len(self._pending_pcm),
                "scored_frames": int(len(self._activity)),
                "frame_step_ms": FRAME_MS,
                "cross_chunk_identity_cache": True,
                "speaker_count": len({
                    row["speaker_id"] for row in rows
                }),
                "turns": rows if self._state == "completed" else [],
                "complete": self._state == "completed",
            }

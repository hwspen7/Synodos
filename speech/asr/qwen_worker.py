"""Persistent isolated Qwen3-ASR CUDA worker for local Meeting IPC.

The protocol is newline-delimited JSON on stdin and stdout. The transcript
never leaves the local machine, and no network listener is exposed.
"""
from __future__ import annotations
import base64
import json
import sys
import traceback
import numpy as np
from speech.asr.qwen import QwenEngine

PREFIX = "MEETING_QWEN_WORKER_V1:"
MODEL = "Qwen/Qwen3-ASR-1.7B"
MAX_SAMPLES = 16000 * 120


def encode_recognition(value):
    return {
        "language": value.language,
        "speech_detected": value.speech_detected,
        "confidence": value.confidence,
        "speech_intervals_ms": value.speech_intervals_ms,
        "utterances": [
            {
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "language": u.language,
                "timestamp_quality": u.timestamp_quality,
                "confidence": u.confidence,
                "language_confidence": u.language_confidence,
                "words": [{"text": w.text, "start_ms": w.start_ms,
                           "end_ms": w.end_ms} for w in u.words],
            }
            for u in value.utterances
        ],
    }


def process_request(engine, request):
    command = request.get("command")
    if command == "prewarm":
        engine._get_model()
        return {"ready": True}
    if command != "recognize":
        raise ValueError("Unsupported Qwen worker command")
    raw = base64.b64decode(request["audio_f32le"], validate=True)
    if len(raw) % 4 != 0 or not 0 < len(raw) // 4 <= MAX_SAMPLES:
        raise ValueError("Invalid bounded Qwen audio size")
    samples = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite Qwen audio")
    result = engine.recognize(
        samples, language=str(request.get("language", "auto")),
        terms=str(request.get("terms", ""))[:1500],
    )
    return encode_recognition(result)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen worker requires CUDA PyTorch")
    engine = QwenEngine(MODEL, device="cuda:0", aligner=False)
    for line in sys.stdin:
        request_id = None
        try:
            command = json.loads(line)
            request_id = command.get("id")
            result = process_request(engine, command)
            reply = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {
                "id": request_id, "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:220],
            }
        # Protocol frames are guaranteed ASCII even for multilingual
        # transcripts. Native libraries can print arbitrary code-page
        # bytes to stdout without corrupting our frame serialization.
        frame = (PREFIX + json.dumps(
            reply, ensure_ascii=True, separators=(",", ":")
        ) + "\n").encode("ascii")
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()

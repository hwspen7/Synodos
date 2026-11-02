"""Safe, opt-in WAV archive for replay, bounded by per-session byte quota."""
from __future__ import annotations
from pathlib import Path
import threading
import uuid
import wave


class RecordingQuotaExceeded(RuntimeError):
    pass


class RecordingWriter:
    def __init__(self, root: str | Path, meeting_id: str, quota: int = 4_000_000_000):
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", meeting_id):
            raise ValueError("Invalid recording meeting ID")
        self.folder = Path(root).resolve() / meeting_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{uuid.uuid4().hex}.wav"
        self.quota = quota
        self.written = 0
        self.lock = threading.RLock()
        self.file = wave.open(str(self.path), "wb")
        self.file.setnchannels(1)
        self.file.setsampwidth(2)
        self.file.setframerate(16000)
        self.closed = False

    def append(self, data: bytes) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("Recording already closed")
            if len(data) % 2:
                raise ValueError("PCM frames must have even byte length")
            if self.written + len(data) > self.quota:
                raise RecordingQuotaExceeded("Recording byte quota exceeded")
            self.file.writeframesraw(data)
            self.written += len(data)

    def close(self):
        with self.lock:
            if not self.closed:
                self.file.close()
                self.closed = True

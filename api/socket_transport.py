"""Validate streaming PCM payloads before sending them to audio buffers."""

from __future__ import annotations

import numpy as np
import struct
import hashlib

MAX_PCM_FRAME_BYTES = 1024 * 1024


def decode_pcm_frame(frame: bytes) -> np.ndarray:
    if not frame:
        raise ValueError("Empty PCM frame")
    if len(frame) % 2:
        raise ValueError("PCM frame length must be even")
    if len(frame) > MAX_PCM_FRAME_BYTES:
        raise ValueError("PCM frame exceeds the maximum size")
    return np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0


# Versioned, ordered PCM frames. The sample offset prevents silent gaps;
# the monotonically increasing sequence makes accidental retries idempotent.
_FRAME_MAGIC = b"MOPA"
_FRAME_HEADER = struct.Struct(">4sIQ")


class AudioSequenceError(ValueError):
    """A frame was missing or arrived at the wrong sample offset."""


class AudioFrameSequence:
    def __init__(self):
        self.next_sequence = 0
        self.samples = 0
        self.versioned = False
        self._recent: dict[int, tuple[int, bytes]] = {}

    def accept(self, data: bytes) -> tuple[np.ndarray | None, bool]:
        """Return (audio, duplicate). Legacy raw PCM remains compatible."""
        if data.startswith(_FRAME_MAGIC):
            if len(data) <= _FRAME_HEADER.size:
                raise AudioSequenceError("Truncated audio frame")
            _, sequence, offset = _FRAME_HEADER.unpack_from(data)
            samples = decode_pcm_frame(data[_FRAME_HEADER.size:])
            if not self.versioned and self.samples:
                raise AudioSequenceError("Cannot mix legacy and sequenced audio")
            self.versioned = True
            fingerprint = hashlib.blake2s(data, digest_size=12).digest()
            if sequence < self.next_sequence:
                if self._recent.get(sequence) == (offset, fingerprint):
                    return None, True
                raise AudioSequenceError("Conflicting or expired audio frame replay")
            if sequence != self.next_sequence or offset != self.samples:
                raise AudioSequenceError(
                    f"Audio frame discontinuity: expected sequence {self.next_sequence} "
                    f"at sample {self.samples}, got {sequence} at {offset}"
                )
            self._recent[sequence] = (offset, fingerprint)
            if len(self._recent) > 256:
                del self._recent[min(self._recent)]
            self.next_sequence += 1
            self.samples += len(samples)
            return samples, False
        if self.versioned:
            raise AudioSequenceError("Unframed audio inside a sequenced recording")
        samples = decode_pcm_frame(data)
        self.samples += len(samples)
        return samples, False

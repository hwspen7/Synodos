"""Versioned audio frames are ordered, idempotent, and gap-free."""
import struct
import unittest

import numpy as np

from api.socket_transport import AudioFrameSequence, AudioSequenceError


def packet(sequence, offset, pcm):
    raw = np.asarray(pcm, dtype="<i2").tobytes()
    return struct.pack(">4sIQ", b"MOPA", sequence, offset) + raw


class AudioFrameV2Tests(unittest.TestCase):
    def test_monotonic_frames_and_duplicate_replay(self):
        stream = AudioFrameSequence()
        first = packet(0, 0, [1, 2, 3])
        second = packet(1, 3, [4, 5])
        chunk, duplicate = stream.accept(first)
        self.assertFalse(duplicate)
        self.assertEqual(len(chunk), 3)
        self.assertEqual(stream.samples, 3)
        chunk, duplicate = stream.accept(first)
        self.assertIsNone(chunk)
        self.assertTrue(duplicate)
        self.assertEqual(stream.samples, 3)
        chunk, duplicate = stream.accept(second)
        self.assertFalse(duplicate)
        self.assertEqual(len(chunk), 2)
        self.assertEqual(stream.samples, 5)
        self.assertEqual(stream.next_sequence, 2)

    def test_gap_and_conflicting_replay_are_fatal(self):
        stream = AudioFrameSequence()
        stream.accept(packet(0, 0, [1, 2]))
        with self.assertRaisesRegex(AudioSequenceError, "discontinuity"):
            stream.accept(packet(2, 2, [3]))
        with self.assertRaisesRegex(AudioSequenceError, "Conflicting"):
            stream.accept(packet(0, 1, [3]))

    def test_legacy_audio_compatible_but_cannot_mix_versions(self):
        stream = AudioFrameSequence()
        chunk, duplicate = stream.accept(np.asarray([1, -1], dtype="<i2").tobytes())
        self.assertFalse(duplicate)
        self.assertEqual(len(chunk), 2)
        with self.assertRaisesRegex(AudioSequenceError, "Cannot mix"):
            stream.accept(packet(0, 0, [2]))
        other = AudioFrameSequence()
        other.accept(packet(0, 0, [1, 2]))
        with self.assertRaisesRegex(AudioSequenceError, "Unframed"):
            other.accept(np.asarray([3], dtype="<i2").tobytes())

    def test_short_header_and_oversize_rejected(self):
        stream = AudioFrameSequence()
        with self.assertRaisesRegex(AudioSequenceError, "Truncated"):
            stream.accept(b"MOPA")
        with self.assertRaises(ValueError):
            stream.accept(packet(0, 0, [0] * (1024 * 1024 // 2 + 1)))


if __name__ == "__main__":
    unittest.main()

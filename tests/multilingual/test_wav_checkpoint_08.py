"""WAV checkpoint recovery tests for long-running meetings."""
import tempfile
import unittest
import wave
from pathlib import Path

from speech.recording import RecordingWriter


class RecordingCheckpointTests(unittest.TestCase):
    def test_header_remains_readable_before_clean_shutdown(self):
        with tempfile.TemporaryDirectory() as root:
            writer = RecordingWriter(root, "meeting_test", quota=200000)
            try:
                writer.append(bytes(32000))
                with wave.open(str(writer.path), "rb") as recording:
                    self.assertEqual(recording.getframerate(), 16000)
                    self.assertEqual(recording.getnframes(), 16000)
                    self.assertEqual(len(recording.readframes(16000)), 32000)
                writer.append(bytes(32000))
            finally:
                writer.close()
            with wave.open(str(writer.path), "rb") as recording:
                self.assertEqual(recording.getnframes(), 32000)

    def test_quota_blocks_overflow(self):
        from speech.recording import RecordingQuotaExceeded
        with tempfile.TemporaryDirectory() as root:
            writer = RecordingWriter(root, "meeting_test", quota=32000)
            try:
                writer.append(bytes(32000))
                with self.assertRaises(RecordingQuotaExceeded):
                    writer.append(bytes(2))
            finally:
                writer.close()

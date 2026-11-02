import unittest
import tempfile
import wave
from pathlib import Path
import numpy as np
from speech.recording import RecordingWriter, RecordingQuotaExceeded
from speech.enhancement import quality
from speech.diarization import SpeakerCentroidTracker

class AudioTests(unittest.TestCase):
    def test_wav_reopens_and_preserves_pcm(self):
        with tempfile.TemporaryDirectory() as root:
            w = RecordingWriter(root, "meeting_A", 1000000)
            w.append(b"\x01\x00\x02\x00")
            path = w.path
            w.close()
            with wave.open(str(path), "rb") as file:
                self.assertEqual(file.getframerate(), 16000)
                self.assertEqual(file.readframes(2), b"\x01\x00\x02\x00")

    def test_invalid_meeting_path(self):
        with self.assertRaises(ValueError):
            RecordingWriter("recordings", "../secret")

    def test_recording_quota(self):
        with tempfile.TemporaryDirectory() as root:
            w = RecordingWriter(root, "valid", quota=2)
            with self.assertRaises(RecordingQuotaExceeded):
                w.append(b"1234")
            w.close()

    def test_quality_clipping(self):
        q = quality(np.asarray([1.0, -1.0, 0.0], np.float32))
        self.assertGreater(q["clipped_ratio"], 0)

    def test_speaker_tracker_multilingual_input(self):
        tracker = SpeakerCentroidTracker(threshold=0.8)
        self.assertEqual(tracker.assign(np.array([1.,0.])), 1)
        self.assertEqual(tracker.assign(np.array([1.,0.])), 1)
        self.assertEqual(tracker.assign(np.array([0.,1.])), 2)
        self.assertIsNone(tracker.assign(None))

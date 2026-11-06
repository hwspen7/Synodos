import unittest
import numpy as np
from sessions.audio_buffer import AudioBuffer, AudioBackpressureError


class AudioPressureTests(unittest.TestCase):
    def test_buffer_never_exceeds_capacity(self):
        buf = AudioBuffer(16000, max_seconds=1)
        buf.append(np.zeros(16000, dtype=np.float32))
        with self.assertRaises(AudioBackpressureError):
            buf.append(np.zeros(1, dtype=np.float32))
        self.assertEqual(buf.total_samples(), 16000)
        buf.drop_before(8000)
        buf.append(np.zeros(8000, dtype=np.float32))
        self.assertEqual(buf.total_samples(), 16000)

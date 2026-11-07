import unittest
import numpy as np
from speech.audio.silence_retention import safe_silence_prefix


class SilenceRetentionTests(unittest.TestCase):
    def test_pure_silence_reclaimed_with_tail(self):
        data = np.zeros(16000*12, dtype=np.float32)
        drop = safe_silence_prefix(data)
        self.assertGreater(drop, 16000*8)
        self.assertGreaterEqual(len(data)-drop, 16000*3)

    def test_voice_in_middle_is_preserved(self):
        data = np.zeros(16000*12, dtype=np.float32)
        data[16000*3:16000*4] = .20
        drop = safe_silence_prefix(data)
        self.assertLessEqual(drop, 16000*3)

    def test_weak_voice_like_sound_is_not_dropped(self):
        data = np.zeros(16000*12, dtype=np.float32)
        data[16000:16000+20] = .08
        self.assertLess(safe_silence_prefix(data), 16000*2)

    def test_continuous_noisy_input_fail_closed(self):
        data = np.ones(16000*12, dtype=np.float32)*.035
        self.assertEqual(safe_silence_prefix(data), 0)

    def test_too_short_not_reclaimed(self):
        self.assertEqual(safe_silence_prefix(np.zeros(16000*5, dtype=np.float32)), 0)

    def test_nonfinite_not_reclaimed(self):
        data = np.zeros(16000*12, dtype=np.float32)
        data[100] = np.nan
        self.assertEqual(safe_silence_prefix(data), 0)

    def test_nonmono_not_reclaimed(self):
        self.assertEqual(safe_silence_prefix(np.zeros((10000, 2), dtype=np.float32)), 0)

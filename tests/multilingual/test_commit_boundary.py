"""Continuous-window acoustic commit boundary regression tests."""
import unittest

from speech.commit_boundary import completed_voice_drop


class AcousticCommitBoundaryTests(unittest.TestCase):
    def test_small_confirmed_voice_tail_released_only_after_silence(self):
        self.assertEqual(
            completed_voice_drop(32000, 16000 * 8, ((950, 2250),)),
            37280,
        )

    def test_unsettled_voice_tail_not_retired(self):
        self.assertEqual(
            completed_voice_drop(32000, 16000 * 2 + 4000, ((950, 2250),)),
            32000,
        )

    def test_large_untranscribed_voice_after_last_word_retained(self):
        self.assertEqual(
            completed_voice_drop(32000, 16000 * 8, ((950, 2700),)),
            32000,
        )

    def test_next_turn_nearby_never_crossed(self):
        self.assertEqual(
            completed_voice_drop(32000, 16000 * 8,
                                 ((950, 2250), (2340, 3100))),
            32000,
        )

    def test_missing_vad_provenance_preserves_cursor(self):
        self.assertEqual(completed_voice_drop(32000, 16000 * 8, ()), 32000)

    def test_invalid_audio_geometry_rejected(self):
        with self.assertRaises(ValueError):
            completed_voice_drop(900, 800, ())


if __name__ == "__main__":
    unittest.main()

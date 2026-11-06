"""Continuous-window acoustic commit boundary regression tests."""
import unittest

from speech.captions.commit_boundary import (
    completed_voice_drop, completed_island_drop, unclosed_voice_turn,
)


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

    def test_complete_island_releases_small_supported_tail(self):
        self.assertEqual(completed_island_drop(
            32000, 16000 * 8, ((950, 2150),),
            published_intervals_ms=((950, 2000),),
        ), 35680)

    def test_incomplete_speech_not_consumed(self):
        self.assertEqual(completed_island_drop(
            32000, 16000 * 8, ((900, 3100),),
            published_intervals_ms=((1000, 2000),),
        ), 32000)

    def test_new_word_or_new_speaker_after_pause_is_preserved(self):
        self.assertEqual(completed_island_drop(
            32000, 16000 * 8, ((950, 2150), (2260, 3650)),
            published_intervals_ms=((950, 2000),),
        ), 32000)

    def test_pending_short_reply_never_discarded(self):
        self.assertEqual(completed_island_drop(
            32000, 16000 * 8, ((950, 2150),),
            published_intervals_ms=((950, 2000),),
            held_intervals_ms=((2050, 2150),),
        ), 32000)

    def test_insufficient_onset_coverage_preserved(self):
        self.assertEqual(completed_island_drop(
            32000, 16000 * 8, ((900, 2750),),
            published_intervals_ms=((1600, 2000),),
        ), 32000)

    def test_active_voice_turn_waits_for_ending(self):
        self.assertTrue(unclosed_voice_turn(
            1000, 1900, 2500, ((900, 2200),),
        ))
        self.assertFalse(unclosed_voice_turn(
            1000, 1900, 7000, ((900, 2200),),
        ))

    def test_continuous_speech_cannot_hold_all_earlier_complete_sentences(self):
        self.assertFalse(unclosed_voice_turn(
            1000, 1900, 8500, ((900, 8500),),
        ))

    def test_invalid_audio_geometry_rejected(self):
        with self.assertRaises(ValueError):
            completed_voice_drop(900, 800, ())


if __name__ == "__main__":
    unittest.main()

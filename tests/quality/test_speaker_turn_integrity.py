"""Safety properties for asynchronously returned speaker/word partitions."""
from __future__ import annotations

import unittest

from speech.turn_integrity import valid_turn_partition


class SpeakerTurnIntegrityTests(unittest.TestCase):
    def test_original_fallback_is_valid(self):
        self.assertTrue(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world.", "emb": None},
        ]))

    def test_multiple_speaker_turns_keep_every_recognized_character(self):
        self.assertTrue(valid_turn_partition(1000, 4800, "Hello, world. Yes!", [
            {"start": 1000, "end": 2799, "text": "Hello, world."},
            {"start": 2800, "end": 4800, "text": " Yes!"},
        ]))

    def test_truncated_text_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 4800, "Hello world. Yes!", [
            {"start": 1000, "end": 2799, "text": "Hello world."},
        ]))

    def test_modified_punctuation_or_hallucinated_word_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world!"},
        ]))
        self.assertFalse(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world. Extra"},
        ]))

    def test_out_of_bounds_or_material_overlap_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 4800, "First second", [
            {"start": 1000, "end": 3000, "text": "First"},
            {"start": 2800, "end": 4700, "text": " second"},
        ]))
        self.assertFalse(valid_turn_partition(1000, 4800, "First second", [
            {"start": 1000, "end": 3000, "text": "First"},
            {"start": 3000, "end": 5100, "text": " second"},
        ]))

    def test_small_rounding_tolerance_is_accepted(self):
        self.assertTrue(valid_turn_partition(1000, 4800, "A B", [
            {"start": 990, "end": 3000, "text": "A"},
            {"start": 2980, "end": 4840, "text": " B"},
        ]))

    def test_malformed_worker_output_never_replaces_text(self):
        for candidate in (None, [], [None], ["text"], [
            {"start": True, "end": 2000, "text": "Hi"},
        ], [{"start": 1000, "end": 1000, "text": "Hi"}], [
            {"start": 1000, "end": 2000, "text": ""},
        ]):
            with self.subTest(candidate=candidate):
                self.assertFalse(valid_turn_partition(1000, 2000, "Hi", candidate))

    def test_nfkc_and_spacing_are_only_allowed_normalizations(self):
        self.assertTrue(valid_turn_partition(0, 2000, "ＡＢ C", [
            {"start": 0, "end": 2000, "text": "AB C"},
        ]))
        self.assertFalse(valid_turn_partition(0, 2000, "A B C", [
            {"start": 0, "end": 2000, "text": "A B D"},
        ]))


if __name__ == "__main__":
    unittest.main()

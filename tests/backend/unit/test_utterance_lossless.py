"""Regression tests for lossless Whisper word-timed sentence partitioning."""
from __future__ import annotations

import unittest

from speech.contracts import Utterance, Word
from speech.text_format import join_transcript_text
from speech.utterance_units import split_utterance, split_utterances


class LosslessUtteranceSplitTests(unittest.TestCase):
    def test_exact_two_sentences_split_with_original_punctuation(self):
        original = "Really, what a coincidence. We laughed a lot."
        words = (
            Word("Really,", 0, 430),
            Word(" what", 460, 710),
            Word(" a", 720, 820),
            Word(" coincidence.", 830, 1510),
            Word(" We", 1680, 1900),
            Word(" laughed", 1910, 2350),
            Word(" a", 2390, 2490),
            Word(" lot.", 2500, 3000),
        )
        source = Utterance(original, 0, 3100, "en", words, "word")
        parts = split_utterance(source)
        self.assertEqual(len(parts), 2)
        self.assertEqual([p.text for p in parts],
                         ["Really, what a coincidence.", "We laughed a lot."])
        self.assertEqual([(p.start_ms, p.end_ms) for p in parts],
                         [(0, 1510), (1680, 3000)])
        reconstructed = ""
        for part in parts:
            reconstructed = join_transcript_text(reconstructed, part.text)
        self.assertEqual(reconstructed, original)

    def test_one_missing_word_cannot_be_discarded_by_close_alignment(self):
        original = ("We carefully reviewed the new release schedule and identified "
                    "a critical dependency. Then we assigned the next steps.")
        words = [
            Word("We", 0, 250),
            Word(" carefully", 280, 520),
            Word(" reviewed", 550, 770),
            Word(" the", 790, 850),
            Word(" new", 880, 950),
            Word(" release", 1000, 1200),
            Word(" schedule", 1230, 1450),
            Word(" and", 1500, 1590),
            Word(" identified", 1630, 1820),
            Word(" a", 1840, 1910),
            # Critical is missing from the alleged word-level alignment.
            Word(" dependency.", 2120, 2400),
            Word(" Then", 2580, 2760),
            Word(" we", 2800, 2920),
            Word(" assigned", 2940, 3180),
            Word(" the", 3200, 3300),
            Word(" next", 3320, 3480),
            Word(" steps.", 3500, 3780),
        ]
        source = Utterance(original, 0, 3900, "en", tuple(words), "word")
        self.assertEqual(split_utterance(source), (source,))

    def test_missing_or_modified_punctuation_cannot_rewrite_original(self):
        original = "We don't know yet. Let's check."
        words = (
            Word("We", 0, 180),
            Word(" dont", 200, 520),
            Word(" know", 560, 810),
            Word(" yet.", 840, 1230),
            Word(" Lets", 1440, 1750),
            Word(" check.", 1780, 2200),
        )
        source = Utterance(original, 0, 2300, "en", words, "word")
        self.assertEqual(split_utterance(source), (source,))
        source2 = Utterance("Okay! Next.", 0, 1100, "en", (
            Word("Okay.", 0, 500), Word(" Next.", 650, 1000),
        ), "word")
        self.assertEqual(split_utterance(source2), (source2,))

    def test_cjk_word_timing_only_splits_at_real_sentence_boundaries(self):
        source = Utterance("你好。你好吗？", 0, 2200, "zh", (
            Word("你好。", 0, 1000), Word("你好吗？", 1100, 2200),
        ), "word")
        parts = split_utterance(source)
        self.assertEqual([p.text for p in parts], ["你好。", "你好吗？"])
        self.assertEqual(len(split_utterances((source,))), 2)

    def test_invalid_time_order_preserves_whisper_original(self):
        source = Utterance("First. Second.", 0, 2500, "en", (
            Word("First.", 1300, 1600), Word(" Second.", 700, 900),
        ), "word")
        self.assertEqual(split_utterance(source), (source,))


if __name__ == "__main__":
    unittest.main()

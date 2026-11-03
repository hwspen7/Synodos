"""Regression tests for transcript joining and conservative temporal deduplication."""
import unittest
from speech.text_format import join_transcript_text
from speech.commit_guard import is_temporal_duplicate


class TranscriptCompositionTests(unittest.TestCase):
    def test_latin_spacing(self):
        self.assertEqual(join_transcript_text("Good morning.", "Everyone."), "Good morning. Everyone.")

    def test_chinese_adjacency(self):
        self.assertEqual(join_transcript_text("大家", "好。"), "大家好。")

    def test_japanese_adjacency(self):
        self.assertEqual(join_transcript_text("こんにちは", "世界"), "こんにちは世界")

    def test_punctuation(self):
        self.assertEqual(join_transcript_text("Hello", ", world"), "Hello, world")


class CommitGuardTests(unittest.TestCase):
    def test_identical_time_and_words(self):
        previous = [{"start_sample": 0, "end_sample": 32000, "text": "Same words."}]
        self.assertTrue(is_temporal_duplicate(0, 32000, "same words.", previous))

    def test_repeated_words_at_different_times_are_preserved(self):
        previous = [{"start_sample": 0, "end_sample": 32000, "text": "Same words."}]
        self.assertFalse(is_temporal_duplicate(40000, 72000, "Same words.", previous))

    def test_simultaneous_different_speech_is_preserved(self):
        previous = [{"start_sample": 0, "end_sample": 32000, "text": "One speaker."}]
        self.assertFalse(is_temporal_duplicate(0, 32000, "Another speaker.", previous))

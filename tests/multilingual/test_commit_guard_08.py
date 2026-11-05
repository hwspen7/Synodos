"""Regression tests for transcript joining and conservative temporal deduplication."""
import unittest
from speech.text_format import join_transcript_text
from speech.commit_guard import is_temporal_duplicate, is_adjacent_tail_echo


class TranscriptCompositionTests(unittest.TestCase):
    def test_latin_spacing(self):
        self.assertEqual(join_transcript_text("Good morning.", "Everyone."), "Good morning. Everyone.")

    def test_chinese_adjacency(self):
        self.assertEqual(join_transcript_text("\u5927\u5bb6", "\u597d\u3002"), "\u5927\u5bb6\u597d\u3002")

    def test_japanese_adjacency(self):
        self.assertEqual(join_transcript_text("\u3053\u3093\u306b\u3061\u306f", "\u4e16\u754c"), "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c")

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

    def test_adjacent_unaligned_tail_word_is_not_committed_twice(self):
        prior = [{
            "start_sample": 40000, "end_sample": 56000,
            "text": "Who are you?",
        }]
        self.assertTrue(is_adjacent_tail_echo(
            56000, 60500, "You.", prior
        ))
        self.assertFalse(is_adjacent_tail_echo(
            56000, 60500, "You.", prior, timestamp_quality="word"
        ))

    def test_genuine_repetition_and_code_switch_are_preserved(self):
        prior = [{
            "start_sample": 40000, "end_sample": 56000,
            "text": "Who are you?",
        }]
        self.assertFalse(is_adjacent_tail_echo(
            62000, 69000, "You.", prior
        ))
        self.assertFalse(is_adjacent_tail_echo(
            55000, 61000, "Yes.", prior
        ))
        self.assertFalse(is_adjacent_tail_echo(
            56000, 60200, "\u94b1\u3002", [{
                "start_sample": 35000, "end_sample": 56000,
                "text": "I'm Spencer Chen.",
            }]
        ))

    def test_full_sentence_echo_and_overlap_other_speaker_preserved(self):
        prior = [{
            "start_sample": 40000, "end_sample": 56000,
            "text": "Who are you?",
        }]
        self.assertFalse(is_adjacent_tail_echo(
            56000, 61000, "Who are you?", prior
        ))
        self.assertFalse(is_adjacent_tail_echo(
            40000, 45000, "You.", prior
        ))

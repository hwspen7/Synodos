import unittest
from speech.contracts import Recognition, Utterance, Word
from speech.asr.streaming import PreviewTracker
from speech.config import SpeechSettings
from speech.language import normalize_language

class ContractTests(unittest.TestCase):
    def test_word_timestamps_remain_separate(self):
        segment = Utterance("hello", 0, 1100, "en", (Word("hello", 100, 990),), "word")
        row = segment.as_legacy()
        self.assertEqual(row["words"][0]["start_ms"], 100)
        self.assertEqual(row["timestamp_quality"], "word")

    def test_preview_revisions_are_monotonic(self):
        previews = PreviewTracker()
        self.assertEqual(previews.update("hello", "en", 0, 1000).revision, 1)
        self.assertIsNone(previews.update("hello", "en", 0, 1000))
        self.assertEqual(previews.update("hello there", "en", 0, 1000).revision, 2)
        previews.clear()
        self.assertIsNotNone(previews.update("hello", "en", 0, 1000))

    def test_settings_reject_invalid_engine(self):
        with self.assertRaises(ValueError):
            SpeechSettings(engine="not_a_model").validate()

    def test_qwen_language_normalization(self):
        self.assertEqual(normalize_language("Chinese"), "zh")
        self.assertEqual(normalize_language("English"), "en")

    def test_valid_settings_allow_disabled_optional_features(self):
        self.assertEqual(SpeechSettings().validate().engine, "whisper")

    def test_native_streaming_needs_online_model(self):
        with self.assertRaises(ValueError):
            SpeechSettings(engine="legacy", native_streaming=True).validate()

import unittest
from tools.evaluation.asr.asr_runtime_benchmark import edit_distance, score


class ASRQualityMetricsTests(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(edit_distance(list("abcd"), list("abcd")), 0)

    def test_empty_hypothesis(self):
        self.assertEqual(edit_distance(list("abc"), []), 3)

    def test_language_accuracy(self):
        result = score([{"reference": "hello world", "hypothesis": "hello world",
                         "reference_language": "en", "detected_language": "en"}])
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["language_accuracy"], 1)

    def test_missing_language_labels(self):
        self.assertIsNone(score([{"reference": "a", "hypothesis": "b"}])["language_accuracy"])

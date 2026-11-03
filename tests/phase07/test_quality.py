"""Phase 07A benchmark metrics."""
import unittest
from phase07.quality import normalize_text, char_tokens, mixed_tokens, edit_distance, score_pair, aggregate


class RecognitionQualityTests(unittest.TestCase):
    def test_unicode_and_punctuation_normalization(self):
        self.assertEqual(normalize_text("\uff21\uff22\uff23\uff0c HELLO\uff01"), "abc hello")

    def test_cjk_character_count(self):
        self.assertEqual(char_tokens("\u4f60\u597d\uff0c World!"), list("\u4f60\u597dworld"))

    def test_mixed_language_tokenization(self):
        self.assertEqual(mixed_tokens("\u4f60\u597d Hello World"), ["\u4f60", "\u597d", "hello", "world"])

    def test_exact_score(self):
        row = score_pair("\u4f60\u597d world", "\u4f60\u597d world")
        self.assertEqual(row["cer"], 0)
        self.assertEqual(row["mixed_token_error_rate"], 0)

    def test_single_character_edit(self):
        self.assertAlmostEqual(score_pair("\u4f60\u597d", "\u4f60\u4eec")["cer"], .5)

    def test_empty_reference_not_division_by_zero(self):
        self.assertEqual(score_pair("", "anything")["reference_characters"], 0)

    def test_levenshtein(self):
        self.assertEqual(edit_distance(list("kitten"), list("sitting")), 3)

    def test_glossary(self):
        result = score_pair("Discuss X-Plane today", "Discuss X-Plane tomorrow", terms=["X-Plane", "unmentioned"])
        self.assertEqual(result["glossary_hits"], 1)
        self.assertEqual(result["glossary_terms"], 1)

    def test_weighted_metrics(self):
        a = score_pair("\u4f60\u597d", "\u4f60")
        b = score_pair("hello", "hello")
        combined = aggregate([a, b])
        self.assertAlmostEqual(combined["cer"], 1 / 7)

    def test_language_accuracy(self):
        item = dict(score_pair("hi", "hi"), expected_language="en", predicted_language="en")
        self.assertEqual(aggregate([item])["language_accuracy"], 1.0)

    def test_failure_count(self):
        self.assertEqual(aggregate([{"error": "NoModel"}])["failed"], 1)

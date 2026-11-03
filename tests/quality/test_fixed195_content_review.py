"""Tests for honest Fixed195 single-primary and multi-secondary ASR review."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.fixed195_content_review import (
    LANGUAGE_ORDER, review_flags, single_summary, multi_summary,
    validate_singles, validate_public_multi,
)


class Fixed195ContentReviewContracts(unittest.TestCase):
    def test_single_scores_use_character_weighted_rate_not_mean_per_clip(self):
        samples = [
            {"id": "fixed195-en-s01", "kind": "single", "language": "en",
             "reused_fixed150_id": "fixed150-009"},
            {"id": "fixed195-en-s02", "kind": "single", "language": "en"},
            {"id": "fixed195-pt-s01", "kind": "single", "language": "pt"},
        ]
        results = {
            "fixed195-en-s01": {
                "character_edits": 20, "reference_characters": 100,
                "latency_ms": 200.0, "hypothesis": "hello",
                "review_flags": [],
            },
            "fixed195-en-s02": {
                "character_edits": 0, "reference_characters": 10,
                "latency_ms": 100.0, "hypothesis": "hi",
                "review_flags": [],
            },
            "fixed195-pt-s01": {"error": "RuntimeError: injected"},
        }
        summary = single_summary(samples, results)
        self.assertAlmostEqual(summary["overall"]["cer"], 20 / 110)
        self.assertEqual(summary["overall"]["requested"], 3)
        self.assertEqual(summary["overall"]["completed"], 2)
        self.assertEqual(summary["overall"]["failed"], 1)
        self.assertEqual(summary["old_56"]["completed"], 1)
        self.assertEqual(summary["new_74"]["completed"], 1)
        self.assertIsNone(summary["macro_cer"])
        self.assertEqual(summary["per_language"]["en"]["completed"], 2)
        self.assertEqual(summary["per_language"]["pt"]["failed"], 1)

    def test_no_multi_reference_means_no_fabricated_accuracy(self):
        items = [{"id": "fixed195-en-m01"}]
        results = {
            "fixed195-en-m01": {
                "hypothesis": "Hello, how are you?", "review_flags": [],
                "speaker_attribution": "unmeasured",
            }
        }
        summary = multi_summary(items, results)
        self.assertEqual(summary["reviewed_with_model"], 1)
        self.assertIsNone(summary["scoreable_cer"])
        self.assertIsNone(summary["scoreable_diarization"])
        self.assertIsNone(summary["human_intelligibility_rating"])

    def test_flags_are_review_prompts_not_claimed_quality_scores(self):
        self.assertIn("empty_transcription", review_flags("", [], 75))
        self.assertIn("possible_repeated_word",
                      review_flags("hello hello, hello", [], 75))
        self.assertIn("long_utterance_check_segmentation",
                      review_flags("hello", [
                          {"text": "a" * 330, "start_ms": 0, "end_ms": 1000}
                      ], 10))
        self.assertNotIn("empty_transcription",
                         review_flags("Hello, this is a complete sentence.",
                                      [], 3))

    def test_singles_must_be_complete_before_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [{"id": "fixed195-en-s01", "kind": "single",
                     "language": "en", "reference": "hello",
                     "audio": "audio/missing.wav", "sha256": "123"}]
            (root / "manifest_single.jsonl").write_text(
                json.dumps(rows[0]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exactly 130"):
                validate_singles(root)

    def test_optional_public_multi_missing_is_legitimate(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(validate_public_multi(Path(td)), [])


if __name__ == "__main__":
    unittest.main()

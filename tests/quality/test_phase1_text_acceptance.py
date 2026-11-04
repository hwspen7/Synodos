"""Human text acceptance tests are model-free and protect raw content."""
from __future__ import annotations

import copy
import unittest

from tools.replay_transcript_score import (
    score_reference, alignment_error_counts, finalized_transcript,
)
from tools.phase1_text_acceptance import evaluate_phase1_text

AUDIO = "a" * 64
REFERENCE = "b" * 64


def successful_replay() -> dict:
    return {
        "source_audio_sha256": AUDIO,
        "reference_text_sha256": REFERENCE,
        "server_closed": True,
        "delivery_matched": True,
        "sent_samples": 160000,
        "audio_received_samples": 160000,
        "errors": [], "flags": [],
        "confirmed_segment_disappearances": 0,
        "confirmed_text_changes": 0,
        "subtitle_completeness": "reference_scored_manual_review_still_required",
        "reference_scoring": {
            "cer": 0.04,
            "wer": 0.10,
            "reference_words": 100,
            "reference_character_deletion_rate": 0.01,
            "reference_word_deletion_rate": 0.02,
        },
    }


class TextAcceptanceTests(unittest.TestCase):
    def test_reference_deletions_and_insertion_have_correct_direction(self):
        for reference, hypothesis, expected in [
            (list("abc"), list("ac"), (1, 0, 0)),
            (list("ac"), list("abc"), (0, 1, 0)),
            (list("abc"), list("adc"), (0, 0, 1)),
            (list(""), list("abc"), (0, 3, 0)),
            (list("abc"), list(""), (3, 0, 0)),
        ]:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                report = alignment_error_counts(reference, hypothesis)
                self.assertEqual((
                    report["deletions"], report["insertions"],
                    report["substitutions"],
                ), expected)

    def test_alignment_budget_fails_closed_without_fake_zero(self):
        self.assertIsNone(alignment_error_counts(
            "abcdefghij", "abcdefghij", max_cells=10,
        ))

    def test_cer_reports_reference_content_deleted_separately(self):
        report = score_reference(
            "The meeting is tomorrow.",
            "The meeting tomorrow.",
        )
        self.assertGreater(report["character_deletions"], 0)
        self.assertGreater(report["word_deletions"], 0)
        self.assertEqual(
            report["word_edits"],
            sum(report[x] for x in (
                "word_deletions", "word_insertions", "word_substitutions",
            )),
        )

    def test_cjk_cer_without_false_whitespace_wer(self):
        # Source and tests remain ASCII; these are CJK Unicode code points.
        reference = "\u4f60\u597d\u4eca\u5929"
        hypothesis = "\u4f60\u597d\u5929"
        report = score_reference(reference, hypothesis)
        self.assertIsNone(report["wer"])
        self.assertIsNone(report["word_deletions"])
        self.assertEqual(report["character_deletions"], 1)

    def test_audio_order_and_missing_ids_not_invented(self):
        state = {"segment_texts": {
            42: " Hello", 9: " there"
        }, "segment_order": [42, 9]}
        self.assertEqual(finalized_transcript(state), "Hello there")
        with self.assertRaises(ValueError):
            finalized_transcript({
                "segment_texts": {42: "Hello", 9: "there"},
                "segment_order": [42, 42],
            })

    def test_human_approved_full_replay_passes_deterministic_gate(self):
        result = evaluate_phase1_text(
            successful_replay(), expected_audio_sha256=AUDIO,
            expected_reference_sha256=REFERENCE,
            human_reference_verified=True,
        )
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["gates"].values()))
        self.assertNotIn("The meeting", str(result))

    def test_unverified_reference_never_passes_even_if_cer_zero(self):
        replay = successful_replay()
        replay["reference_scoring"]["cer"] = 0.0
        self.assertFalse(evaluate_phase1_text(
            replay, expected_audio_sha256=AUDIO,
            expected_reference_sha256=REFERENCE,
        )["pass"])

    def test_text_revision_or_disappearance_fails(self):
        for key in ("confirmed_text_changes",
                    "confirmed_segment_disappearances"):
            with self.subTest(key=key):
                replay = successful_replay()
                replay[key] = 1
                result = evaluate_phase1_text(
                    replay, expected_audio_sha256=AUDIO,
                    expected_reference_sha256=REFERENCE,
                    human_reference_verified=True,
                )
                self.assertFalse(result["pass"])

    def test_missing_alignment_breakdown_fails_closed(self):
        replay = successful_replay()
        replay["reference_scoring"]["reference_word_deletion_rate"] = None
        self.assertFalse(evaluate_phase1_text(
            replay, expected_audio_sha256=AUDIO,
            expected_reference_sha256=REFERENCE,
            human_reference_verified=True,
        )["pass"])

    def test_inconsistent_audio_or_reference_rejected(self):
        for key in ("source_audio_sha256", "reference_text_sha256"):
            replay = successful_replay()
            replay[key] = "c" * 64
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "checksum mismatch",
            ):
                evaluate_phase1_text(
                    replay, expected_audio_sha256=AUDIO,
                    expected_reference_sha256=REFERENCE,
                )

    def test_incomplete_pcm_or_missing_reference_score_fails(self):
        for modification in (
            {"server_closed": False},
            {"audio_received_samples": 159000},
            {"flags": ["confirmed_segment_disappearance"]},
            {"reference_scoring": None},
        ):
            with self.subTest(modification=modification):
                replay = successful_replay()
                replay.update(modification)
                result = evaluate_phase1_text(
                    replay, expected_audio_sha256=AUDIO,
                    expected_reference_sha256=REFERENCE,
                    human_reference_verified=True,
                )
                self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()

"""Offline regressions for AMI reference screening, not acceptance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.diagnostics.evaluate_meeting_audio_quality import (
    SCHEMA, tokenize, edit_distance, cross_talk_count, evaluate_windows,
    verify_reference, save_report,
)


class AudioQualityScreeningTests(unittest.TestCase):
    def test_words_and_levenshtein(self):
        self.assertEqual(tokenize("Hello, WORLD! It's ready."), ["hello", "world", "it's", "ready"])
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)

    def test_distinct_speaker_overlap(self):
        words = [
            {"speaker": "A", "start_ms": 10, "end_ms": 300},
            {"speaker": "B", "start_ms": 200, "end_ms": 400},
        ]
        self.assertEqual(cross_talk_count(words), 2)
        words[1]["speaker"] = "A"
        self.assertEqual(cross_talk_count(words), 0)

    def test_reference_never_automatically_becomes_gold(self):
        ref = dict(schema="meeting-official-reference-words",
                   meeting="EN2002a", source_audio_sha256="a" * 64,
                   annotation_archive_sha256="b" * 64,
                   independently_reviewed_against_mixed_audio=False,
                   is_qa_gold=False, is_machine_prediction=False,
                   words=[dict(text="test", speaker="A",
                               start_ms=i * 100, end_ms=i * 100 + 50)
                          for i in range(101)])
        self.assertEqual(len(verify_reference(ref, "EN2002a", "a" * 64,
                                              "b" * 64, 20)), 101)
        ref["is_qa_gold"] = True
        with self.assertRaises(ValueError):
            verify_reference(ref, "EN2002a", "a" * 64, "b" * 64, 20)

    def test_clean_speech_screening_counts_real_edits(self):
        ref = [dict(text=w, speaker="A", start_ms=i * 500,
                    end_ms=i * 500 + 150)
               for i, w in enumerate("we will build a better test now".split())]
        pred = [dict(id=0, text="we will build a great test now",
                     start_ms=0, end_ms=3500)]
        report, rows = evaluate_windows("EN2002a", 30, ref, pred, 30)
        self.assertEqual(report["low_overlap_reference_words"], 7)
        self.assertEqual(report["low_overlap_word_edits"], 1)
        self.assertEqual(rows[0]["review_status"], "UNREVIEWED")

    def test_overlap_excluded_from_pooled_score(self):
        ref = [dict(text="a", speaker="A", start_ms=100, end_ms=900),
               dict(text="b", speaker="B", start_ms=200, end_ms=800)] + [
               dict(text="word", speaker="A", start_ms=1000 + i * 300,
                    end_ms=1100 + i * 300) for i in range(7)]
        report, rows = evaluate_windows("EN2002a", 30, ref, [], 30)
        self.assertEqual(report["low_overlap_window_count"], 0)
        self.assertIsNone(report["low_overlap_screening_wer"])
        self.assertEqual(rows[0]["priority"], "OVERLAP_REVIEW")

    def test_zero_asr_counts_as_missing_not_pass(self):
        ref = [dict(text="missing", speaker="A", start_ms=i * 500,
                    end_ms=i * 500 + 100) for i in range(8)]
        report, rows = evaluate_windows("EN2002a", 30, ref, [], 30)
        self.assertEqual(report["low_overlap_screening_wer"], 1.0)
        self.assertEqual(rows[0]["priority"], "MISSING_ASR")

    def test_short_windows_rescue_unoverlapped_speech(self):
        # One short cross-talking event should not discard 20 clean seconds.
        ref = [
            dict(text="first", speaker="A", start_ms=100, end_ms=550),
            dict(text="second", speaker="B", start_ms=250, end_ms=650),
        ]
        for block in (10, 20):
            ref.extend(
                dict(text="word", speaker="A",
                     start_ms=block * 1000 + i * 900,
                     end_ms=block * 1000 + i * 900 + 200)
                for i in range(6)
            )
        small, _ = evaluate_windows("EN2002a", 30, ref, [], 10)
        large, _ = evaluate_windows("EN2002a", 30, ref, [], 30)
        self.assertEqual(small["reference_token_count"], 14)
        self.assertEqual(small["low_overlap_reference_words"], 12)
        self.assertEqual(small["low_overlap_window_count"], 2)
        self.assertEqual(large["low_overlap_reference_words"], 0)
        self.assertIsNone(large["low_overlap_screening_wer"])

    def test_review_outputs_unverified(self):
        with TemporaryDirectory() as td:
            summary, sheet = save_report(Path(td),
                dict(schema=SCHEMA, qa_gold_ready=False, product_release_pass=False),
                [dict(meeting="EN2002a", review_status="UNREVIEWED")])
            self.assertFalse(json.loads(summary.read_text())["qa_gold_ready"])
            self.assertIn("UNREVIEWED", sheet.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()

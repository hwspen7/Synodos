"""Real-time replay transcript scoring is local, reference-backed and opt-in."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from tools.live_caption_ws_replay import capture_event, new_capture
from tools.replay_transcript_score import (
    edit_distance, finalized_transcript, score_reference,
    write_private_transcript,
)


class ReferenceScoringTests(unittest.TestCase):
    def test_final_audio_order_not_segment_id_order(self):
        state = new_capture()
        capture_event(state, {
            "type": "transcript", "version": 1, "segments": [
                {"id": 8, "start_ms": 1000, "end_ms": 1600,
                 "text": "We laughed.", "speaker_id": 2},
                {"id": 4, "start_ms": 1700, "end_ms": 2100,
                 "text": "Exactly.", "speaker_id": 1},
            ],
        }, 7000)
        capture_event(state, {
            "type": "transcript", "version": 2, "segments": [
                {"id": 8, "start_ms": 1000, "end_ms": 1600,
                 "text": "We laughed.", "speaker_id": 2},
                {"id": 12, "start_ms": 1650, "end_ms": 1680,
                 "text": "Yes.", "speaker_id": -1},
                {"id": 4, "start_ms": 1700, "end_ms": 2100,
                 "text": "Exactly.", "speaker_id": 1},
            ],
        }, 8000)
        self.assertEqual(
            finalized_transcript(state), "We laughed. Yes. Exactly.")
        self.assertEqual(state["segment_order"], [8, 12, 4])
        self.assertNotIn("We laughed.", str({
            "segment_order": state["segment_order"],
            "final_segment_count": state["final_segment_count"],
        }))

    def test_exact_reference_returns_real_zero_errors(self):
        result = score_reference(
            "Really, what a coincidence. We laughed!",
            "Really, what a coincidence. We laughed!")
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["reference_words"], 6)
        self.assertNotIn("coincidence", str(result))

    def test_missing_word_and_extra_word_count_as_edits(self):
        result = score_reference("We had a good meeting.",
                                 "We had good meeting again.")
        self.assertGreater(result["cer"], 0)
        self.assertEqual(result["word_edits"], 2)
        self.assertEqual(result["reference_words"], 5)
        self.assertAlmostEqual(result["wer"], 0.4)
        self.assertEqual(result["wer_percent"], 40.0)

    def test_punctuation_only_error_is_reported_separately(self):
        result = score_reference("Hello. Next?", "Hello, Next!")
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)
        self.assertGreater(result["punctuation_sensitive_cer_percent"], 0)

    def test_cjk_cer_without_invalid_whitespace_wer(self):
        result = score_reference("\u4f60\u597d\u3002\u4e16\u754c\uff01",
                                 "\u4f60\u597d\u3002\u4e16\u754c\u3002")
        self.assertEqual(result["cer"], 0)
        self.assertIsNone(result["wer"])
        self.assertGreater(result["punctuation_sensitive_cer_percent"], 0)

    def test_empty_or_invalid_reference_is_not_scored(self):
        with self.assertRaisesRegex(ValueError, "no scoreable"):
            score_reference("...", "Hello")
        with self.assertRaises(TypeError):
            score_reference(None, "Hello")
        self.assertEqual(edit_distance("abc", "ac"), 1)

    def test_transcript_export_only_in_gitignored_root_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Path.cwd()
            os.chdir(temp)
            try:
                target = Path("benchmark/fixed195/replay_audit.txt")
                write_private_transcript(target, "Private spoken words")
                self.assertEqual(target.read_text(encoding="utf-8"),
                                 "Private spoken words\n")
                with self.assertRaises(FileExistsError):
                    write_private_transcript(target, "Cannot overwrite")
                with self.assertRaisesRegex(ValueError, "gitignored"):
                    write_private_transcript(Path("meeting.txt"), "Private")
                with self.assertRaisesRegex(ValueError, "gitignored"):
                    write_private_transcript(
                        Path("benchmark/fixed195/../not_private.txt"),
                        "Private")
                with self.assertRaisesRegex(ValueError, "gitignored"):
                    write_private_transcript(
                        Path("benchmark/fixed195/replay.json"), "Private")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()

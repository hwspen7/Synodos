"""Overlapping voice is risk evidence, never synthetic transcript content."""
from __future__ import annotations

import unittest

from speech.diarization.overlap_content_risk import overlap_content_risk


class OverlapContentRiskTests(unittest.TestCase):
    def test_second_voice_without_parallel_caption_is_reported(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 400, "end_ms": 750, "speaker_id": 2},
        ]
        captions = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
        ]
        report = overlap_content_risk(turns, captions)
        self.assertEqual(report["overlap_acoustic_clock_ms"], 350)
        self.assertEqual(report["overlap_clock_underrepresented_ms"], 350)
        self.assertEqual(report["overlap_clock_no_caption_ms"], 0)
        self.assertEqual(report["review_spans"][0]["speaker_ids"], [1, 2])

    def test_two_parallel_confirmed_caption_speakers_cover_activity(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 400, "end_ms": 750, "speaker_id": 2},
        ]
        captions = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 400, "end_ms": 750, "speaker_id": 2},
        ]
        report = overlap_content_risk(turns, captions)
        self.assertEqual(report["overlap_acoustic_clock_ms"], 350)
        self.assertEqual(report["overlap_clock_underrepresented_ms"], 0)

    def test_no_captions_reports_uncovered_parallel_speech(self):
        turns = [
            {"start_ms": 0, "end_ms": 500, "speaker_id": 1},
            {"start_ms": 150, "end_ms": 350, "speaker_id": 2},
        ]
        report = overlap_content_risk(turns, [])
        self.assertEqual(report["overlap_clock_no_caption_ms"], 200)
        self.assertEqual(report["overlap_clock_underrepresented_ms"], 0)

    def test_partial_above_threshold_does_not_merge_across_silence(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 100, "end_ms": 300, "speaker_id": 2},
            {"start_ms": 700, "end_ms": 900, "speaker_id": 2},
        ]
        report = overlap_content_risk(turns, [])
        self.assertEqual(report["overlap_acoustic_clock_ms"], 400)
        self.assertEqual(
            [(r["start_ms"], r["end_ms"]) for r in report["review_spans"]],
            [(100, 300), (700, 900)],
        )

    def test_invalid_turn_is_never_silently_accepted(self):
        with self.assertRaises(ValueError):
            overlap_content_risk([
                {"start_ms": 0, "end_ms": 500, "speaker_id": 9},
            ], [])


if __name__ == "__main__":
    unittest.main()

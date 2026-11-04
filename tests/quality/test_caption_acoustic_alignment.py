"""Bounded, non-destructive caption-to-Nemotron acoustic support.

The source transcript and its timestamps remain intact. The support
intervals only describe where a confirmed speaker was acoustically active.
"""
from __future__ import annotations

import unittest
import numpy as np

from speech.caption_acoustic_alignment import (
    speaker_support_intervals, caption_support_summary,
)
from tests.quality.test_nemotron_processor_fusion import (
    FakeAcoustics, processor_for_test,
)


class CaptionAcousticAlignmentTests(unittest.TestCase):
    def test_intersection_preserves_multiple_voiced_islands(self):
        turns = [
            {"start_ms": 100, "end_ms": 300, "speaker_id": 1},
            {"start_ms": 350, "end_ms": 450, "speaker_id": 1},
            {"start_ms": 260, "end_ms": 400, "speaker_id": 2},
            {"start_ms": 400, "end_ms": 600, "speaker_id": 1},
        ]
        self.assertEqual(
            speaker_support_intervals(turns, 200, 500, 1),
            [[200, 300], [350, 500]],
        )
        summary = caption_support_summary(turns, 200, 500, 1)
        self.assertEqual(summary["speaker_activity_intervals_ms"],
                         [[200, 300], [350, 500]])
        self.assertEqual(summary["speaker_activity_fraction"], round(250/300, 5))

    def test_overlaps_are_not_collapsed_into_single_speaker(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 500, "end_ms": 900, "speaker_id": 2},
        ]
        one = speaker_support_intervals(turns, 0, 1000, 1)
        two = speaker_support_intervals(turns, 0, 1000, 2)
        self.assertEqual(one, [[0, 1000]])
        self.assertEqual(two, [[500, 900]])

    def test_incomplete_turn_input_is_rejected(self):
        with self.assertRaises(ValueError):
            speaker_support_intervals([
                {"start_ms": 0, "end_ms": 1000, "speaker_id": -3},
            ], 0, 1000, 1)

    def test_official_turn_channel_range_is_strict(self):
        for invalid_id in (-3, 0, 9, True, "1", None):
            with self.subTest(speaker_id=invalid_id):
                with self.assertRaises(ValueError):
                    speaker_support_intervals([
                        {"start_ms": 0, "end_ms": 1000,
                         "speaker_id": invalid_id},
                    ], 0, 1000, 1)

    def test_final_session_publishes_support_without_mutating_asr_bounds(self):
        class Completed(FakeAcoustics):
            def is_complete(self):
                return True
            def final_turns_snapshot(self):
                return [
                    {"start_ms": 200, "end_ms": 450, "speaker_id": 2},
                    {"start_ms": 550, "end_ms": 900, "speaker_id": 2},
                ]

        processor = processor_for_test()
        original_text = processor.segments[0]["text"]
        original_times = (
            processor.segments[0]["start_sample"],
            processor.segments[0]["end_sample"],
        )
        activity = np.zeros((100, 8), dtype=np.float32)
        processor.independent_diarization = Completed(activity)
        self.assertTrue(processor._rebuild_and_push())
        support = processor._independent_support[1]
        self.assertEqual(
            support["speaker_activity_intervals_ms"],
            [[200, 450], [550, 900]],
        )
        self.assertEqual(
            processor.snapshot_state()["segments"][0]["speaker_activity_fraction"],
            0.6,
        )
        self.assertEqual(processor.segments[0]["text"], original_text)
        self.assertEqual(
            (
                processor.segments[0]["start_sample"],
                processor.segments[0]["end_sample"],
            ), original_times,
        )


if __name__ == "__main__":
    unittest.main()

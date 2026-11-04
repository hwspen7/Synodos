"""Model-free paired caption coverage diagnosis.

No transcript text, voice embeddings, CUDA or reference RTTM modification.
"""
from __future__ import annotations

import unittest

from tools.ami_paired_caption_coverage import (
    _caption_intervals, framewise_caption_change,
)
from sessions.speaker_stability import anonymous_speaker_timeline


class PairedCoverageTests(unittest.TestCase):
    def test_simultaneous_new_and_lost_coverage_are_separate(self):
        reference = [{"start_ms": 0, "end_ms": 1000, "speaker": "A"}]
        baseline = [[0, 600]]
        candidate = [[200, 800]]
        result = framewise_caption_change(
            reference, baseline, candidate, duration_ms=1000,
        )
        self.assertEqual(result["reference_clock_s"], 1.0)
        self.assertEqual(result["baseline_covered_s"], 0.6)
        self.assertEqual(result["candidate_covered_s"], 0.6)
        self.assertEqual(result["newly_covered_s"], 0.2)
        self.assertEqual(result["coverage_lost_s"], 0.2)
        self.assertEqual(result["net_new_coverage_s"], 0)
        self.assertEqual(result["longest_newly_covered_intervals"], [
            {"start_ms": 600, "end_ms": 800, "duration_ms": 200},
        ])
        self.assertEqual(result["longest_coverage_lost_intervals"], [
            {"start_ms": 0, "end_ms": 200, "duration_ms": 200},
        ])
        self.assertEqual(result["longest_missing_in_both_intervals"], [
            {"start_ms": 800, "end_ms": 1000, "duration_ms": 200},
        ])

    def test_reference_silence_terminates_missing_runs(self):
        reference = [
            {"start_ms": 0, "end_ms": 300, "speaker": "A"},
            {"start_ms": 500, "end_ms": 800, "speaker": "B"},
        ]
        result = framewise_caption_change(
            reference, [], [], duration_ms=1000,
        )
        self.assertEqual(result["unchanged_missing_s"], 0.6)
        self.assertEqual(
            result["longest_missing_in_both_intervals"],
            [
                {"start_ms": 0, "end_ms": 300, "duration_ms": 300},
                {"start_ms": 500, "end_ms": 800, "duration_ms": 300},
            ],
        )

    def test_overlapping_reference_talkers_count_one_clock_frame(self):
        reference = [
            {"start_ms": 0, "end_ms": 400, "speaker": "A"},
            {"start_ms": 200, "end_ms": 600, "speaker": "B"},
        ]
        result = framewise_caption_change(
            reference, [], [[200, 400]], duration_ms=1000,
        )
        self.assertEqual(result["reference_clock_s"], 0.6)
        self.assertEqual(result["newly_covered_s"], 0.2)
        self.assertEqual(result["net_new_coverage_s"], 0.2)

    def test_final_timeline_provenance_counts_distinct_recovered_segments(self):
        timeline = anonymous_speaker_timeline([
            {
                "id": 1, "start_sample": 0, "end_sample": 1600,
                "speaker_id": 1, "recognition_origin": "primary",
            },
            {
                "id": 2, "start_sample": 1600, "end_sample": 3200,
                "speaker_id": 1, "recognition_origin": "gap_recovery",
            },
        ], 16000)
        self.assertEqual(
            [row["recovery_origin"] for row in timeline["segments"]],
            [False, True],
        )
        self.assertFalse(timeline["truncated"])
        self.assertNotIn("text", str(timeline))

    def test_incomplete_timeline_rejected(self):
        data = {
            "final_cluster_diagnostics": {
                "anonymous_timeline": {
                    "truncated": True, "total_segments": 1,
                    "segments": [{"start_ms": 0, "end_ms": 100}],
                },
            },
        }
        with self.assertRaises(ValueError):
            _caption_intervals(data)

    def test_caption_timeline_requires_exact_integer_geometry(self):
        for item in (
            {"start_ms": 0, "end_ms": 0},
            {"start_ms": -100, "end_ms": 100},
            {"start_ms": 100, "end_ms": 240100},
            {"start_ms": 100.0, "end_ms": 200},
        ):
            with self.subTest(item=item):
                replay = {
                    "final_cluster_diagnostics": {
                        "anonymous_timeline": {
                            "truncated": False, "total_segments": 1,
                            "segments": [item],
                        },
                    },
                }
                with self.assertRaises(ValueError):
                    _caption_intervals(replay)


if __name__ == "__main__":
    unittest.main()

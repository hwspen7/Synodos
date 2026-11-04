"""Model-free paired caption coverage diagnosis.

No transcript text, voice embeddings, CUDA or reference RTTM modification.
"""
from __future__ import annotations

import unittest

from tools.ami_paired_caption_coverage import (
    _caption_intervals, framewise_caption_change,
)


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

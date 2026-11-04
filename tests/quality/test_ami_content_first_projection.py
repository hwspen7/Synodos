"""Projection DER is not canonical published DER or a text recovery metric."""
from __future__ import annotations

import unittest

from tools.ami_nemotron_reference_diagnostic import display_projection_turns
from tools.phase09_speaker_benchmark import evaluate


class DisplayProjectionTests(unittest.TestCase):
    def test_split_cues_replace_one_caption_without_duplicate_parent(self):
        replay = {
            "final_cluster_diagnostics": {
                "anonymous_timeline": {
                    "truncated": False,
                    "total_segments": 2,
                    "segments": [
                        {
                            "start_ms": 0, "end_ms": 1000,
                            "published_speaker_id": 1,
                            "caption_projection_status": "split_verified",
                            "caption_projection_turns": [
                                {"start_ms": 0, "end_ms": 420,
                                 "speaker_id": 1, "word_count": 2},
                                {"start_ms": 600, "end_ms": 960,
                                 "speaker_id": 2, "word_count": 2},
                            ],
                        },
                        {
                            "start_ms": 1100, "end_ms": 1300,
                            "published_speaker_id": 2,
                            "caption_projection_status": "unaligned",
                            "caption_projection_turns": [],
                        },
                    ],
                },
            },
        }
        turns, count = display_projection_turns(replay, 2000)
        self.assertEqual(count, 1)
        self.assertEqual(turns, [
            {"start_ms": 0, "end_ms": 420, "speaker_id": 1},
            {"start_ms": 600, "end_ms": 960, "speaker_id": 2},
            {"start_ms": 1100, "end_ms": 1300, "speaker_id": 2},
        ])
        reference = [
            {"start_ms": 0, "end_ms": 500, "speaker": "a"},
            {"start_ms": 600, "end_ms": 1300, "speaker": "b"},
        ]
        result = evaluate(reference, turns, step_ms=100)
        self.assertEqual(result["reference_speaker_seconds"], 1.2)

    def test_no_verified_splits_can_be_reported_as_no_projection(self):
        record = {
            "final_cluster_diagnostics": {
                "anonymous_timeline": {
                    "truncated": False, "total_segments": 1,
                    "segments": [
                        {"start_ms": 0, "end_ms": 100,
                         "published_speaker_id": 1},
                    ],
                },
            },
        }
        turns, count = display_projection_turns(record, 1000)
        self.assertEqual(count, 0)
        self.assertEqual(len(turns), 1)

    def test_invalid_projected_cue_must_not_be_scored(self):
        record = {
            "final_cluster_diagnostics": {
                "anonymous_timeline": {
                    "truncated": False, "total_segments": 1,
                    "segments": [
                        {
                            "start_ms": 0, "end_ms": 1000,
                            "published_speaker_id": 1,
                            "caption_projection_status": "split_verified",
                            "caption_projection_turns": [
                                {"start_ms": 0, "end_ms": 400,
                                 "speaker_id": 1},
                                {"start_ms": 200, "end_ms": 2000,
                                 "speaker_id": 2},
                            ],
                        },
                    ],
                },
            },
        }
        with self.assertRaises(ValueError):
            display_projection_turns(record, 1000)


if __name__ == "__main__":
    unittest.main()

"""No-download tests for independently scored multi-speaker Nemotron windows."""
from __future__ import annotations

import unittest
import numpy as np

from tools.evaluation.ami.ami_shadow10_nemotron_compare import clip_turns, compare_windows


class FakeDiarizer:
    model_id = "fake-nemotron4"

    def diarize(self, audio):
        # Independent windows intentionally restart anonymous speaker slots.
        return [
            {"start_ms": 0, "end_ms": 40000, "speaker_id": 1},
            {"start_ms": 40000, "end_ms": 80000, "speaker_id": 2},
            {"start_ms": 80000, "end_ms": 120000, "speaker_id": 3},
        ]


class OfflineCompareTests(unittest.TestCase):
    def test_crop_preserves_simultaneous_turns_and_offsets(self):
        rows = [
            {"start_ms": 118000, "end_ms": 123000, "speaker": "A"},
            {"start_ms": 120000, "end_ms": 124000, "speaker": "B"},
        ]
        self.assertEqual(clip_turns(rows, 120000, 120000, "speaker"), [
            {"start_ms": 0, "end_ms": 3000, "speaker": "A"},
            {"start_ms": 0, "end_ms": 4000, "speaker": "B"},
        ])

    def test_scores_four_or_more_speakers_without_hardcoded_two(self):
        reference = []
        baseline = []
        for index in range(2):
            start = index * 120000
            for speaker in range(4):
                reference.append({
                    "start_ms": start + speaker * 30000,
                    "end_ms": start + (speaker + 1) * 30000,
                    "speaker": f"person_{speaker}",
                })
                baseline.append({
                    "start_ms": start + speaker * 30000,
                    "end_ms": start + (speaker + 1) * 30000,
                    "speaker_id": 1 if speaker < 2 else 2,
                })
        windows = [
            np.zeros(120 * 16000, dtype=np.float32),
            np.zeros(120 * 16000, dtype=np.float32),
        ]
        report = compare_windows(reference, baseline, windows, FakeDiarizer())
        self.assertEqual(report["window_count"], 2)
        self.assertFalse(report["cross_window_identity_evaluable"])
        self.assertTrue(report["not_live_equivalent"])
        self.assertEqual(report["windows"][0]["reference_speakers"], 4)
        self.assertGreater(
            report["windows"][0]["baseline"]["der"], 0,
        )
        self.assertGreater(
            report["windows"][0]["nemotron3"]["der"], 0,
        )

    def test_incomplete_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_windows([], [], [np.zeros(16000)], FakeDiarizer())


if __name__ == "__main__":
    unittest.main()

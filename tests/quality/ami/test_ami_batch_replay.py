"""Batch AMI aggregate metrics stay speaker-time weighted and fail closed."""
from __future__ import annotations

import unittest

from tools.evaluation.ami.ami_shadow10_batch_replay import aggregate


class BatchReplayTests(unittest.TestCase):
    def test_real_reference_time_weighted_aggregation(self):
        cases = [
            {
                "meeting": "A", "metrics": {
                    "reference_speaker_seconds": 100, "der": 0.1,
                    "missed_speaker_time": 0.05, "false_alarm_time": 0.03,
                    "confusion_time": 0.02, "reference_speaker_count": 2,
                    "predicted_speaker_count": 2,
                },
                "unidentified": {"count": 5},
                "reference_speech_without_captions": {
                    "reference_speech_without_captions_seconds": 2.0,
                },
            },
            {
                "meeting": "B", "metrics": {
                    "reference_speaker_seconds": 300, "der": 0.4,
                    "missed_speaker_time": 0.2, "false_alarm_time": 0.1,
                    "confusion_time": 0.1, "reference_speaker_count": 6,
                    "predicted_speaker_count": 3,
                },
                "unidentified": {"count": 9},
                "reference_speech_without_captions": {
                    "reference_speech_without_captions_seconds": 11.0,
                },
            },
        ]
        report = aggregate(cases)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["total_reference_speaker_seconds"], 400)
        self.assertEqual(report["weighted_metrics"]["der"], 0.325)
        self.assertEqual(report["weighted_metrics"]["missed_speaker_time"], 0.1625)
        self.assertEqual(report["sum_unidentified_segments"], 14)
        self.assertEqual(report["reference_max_speaker_count"], 6)
        self.assertEqual(report["reported"][1]["predicted_speaker_count"], 3)

    def test_empty_dataset_not_accepted_as_success(self):
        with self.assertRaises(ValueError):
            aggregate([])


if __name__ == "__main__":
    unittest.main()

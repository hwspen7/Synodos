"""Deterministic contract checks for overlap-conditioned AMI DER audit."""
from __future__ import annotations

import unittest

from tools.evaluation.asr.temporal_overlap import stratify


class TemporalOverlapAuditTests(unittest.TestCase):
    def test_one_missing_overlap_speaker_does_not_count_as_solo_miss(self):
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
        ]
        hypothesis = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 1000, "end_ms": 1500, "speaker_id": 2},
        ]
        result = stratify(reference, hypothesis, duration_ms=1500, bin_ms=500)
        self.assertEqual(result["reference_speaker_seconds"], 2.0)
        self.assertEqual(result["overall"]["der"], .25)
        self.assertEqual(result["conditions"]["single"]["missed_speaker_seconds"], 0)
        self.assertEqual(result["conditions"]["overlap"]["missed_speaker_seconds"], .5)
        self.assertEqual(result["conditions"]["overlap"]["reference_speaker_seconds"], 1.0)
        self.assertEqual(result["share_of_missed_speaker_time_during_overlap"], 1.0)
        self.assertEqual(result["conditions"]["overlap"]["overlap_underactive_fraction"], 1.0)
        self.assertEqual(result["conditions"]["overlap"]["all_correct_overlap_fraction"], 0)
        self.assertEqual(len(result["worst_30_second_windows"]), 3)
        self.assertNotIn("speaker_mapping", result)

    def test_detects_exact_overlap_with_two_predicted_speakers(self):
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
        ]
        hypothesis = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 500, "end_ms": 1500, "speaker_id": 2},
        ]
        result = stratify(reference, hypothesis, duration_ms=1500)
        self.assertEqual(result["overall"]["der"], 0)
        self.assertEqual(result["conditions"]["overlap"]["all_correct_overlap_fraction"], 1.0)
        self.assertEqual(result["conditions"]["overlap"]["overlap_underactive_fraction"], 0)
        self.assertIsNone(result["share_of_missed_speaker_time_during_overlap"])

    def test_silence_false_alarm_has_no_reference_speaker_denominator(self):
        reference = [
            {"start_ms": 0, "end_ms": 500, "speaker": "A"},
        ]
        hypothesis = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
        ]
        result = stratify(reference, hypothesis, duration_ms=1000)
        self.assertEqual(result["overall"]["der"], 1.0)
        self.assertEqual(result["conditions"]["silence"]["false_alarm_speaker_seconds"], .5)
        self.assertEqual(result["conditions"]["silence"]["reference_speaker_seconds"], 0)
        self.assertIsNone(result["conditions"]["silence"]["der"])

    def test_global_mapping_not_recomputed_for_each_condition(self):
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
            {"start_ms": 1000, "end_ms": 1500, "speaker": "A"},
        ]
        hypothesis = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 1000, "end_ms": 2000, "speaker_id": 2},
        ]
        result = stratify(reference, hypothesis, duration_ms=2000)
        self.assertEqual(result["overall"]["der"], .2)
        self.assertEqual(result["conditions"]["overlap"]["missed_speaker_seconds"], .5)
        self.assertEqual(result["conditions"]["single"]["missed_speaker_seconds"], 0)

    def test_bad_bounds_and_grid_never_approve(self):
        reference = [{"start_ms": 0, "end_ms": 1000, "speaker": "A"}]
        predicted = [{"start_ms": 0, "end_ms": 1000, "speaker_id": 1}]
        with self.assertRaisesRegex(ValueError, "grid"):
            stratify(reference, predicted, duration_ms=1020)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            stratify(reference, predicted, duration_ms=500)


if __name__ == "__main__":
    unittest.main()

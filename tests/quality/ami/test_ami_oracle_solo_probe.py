"""No-model tests for human-reference solo AMI speaker embedding probes."""
from __future__ import annotations

import json
import unittest

import numpy as np

from tools.evaluation.ami.ami_oracle_solo_probe import (
    analyze_solo, reference_solo_runs, select_solo_windows,
    scalar_similarities,
)


class FakeEmbedding:
    def embed(self, samples: np.ndarray):
        value = float(np.mean(samples))
        vector = np.zeros(64, dtype=np.float32)
        if 0.08 < value < 0.14:
            vector[0] = 1
        elif 0.18 < value < 0.24:
            vector[1] = 1
        elif 0.28 < value < 0.34:
            vector[2] = 1
        else:
            return None
        return vector


class OracleSoloProbeTests(unittest.TestCase):
    def test_human_reference_solo_runs_exclude_every_overlap(self):
        reference = [
            {"start_ms": 0, "end_ms": 10000, "speaker": "A"},
            {"start_ms": 2000, "end_ms": 6000, "speaker": "B"},
            {"start_ms": 4000, "end_ms": 5000, "speaker": "C"},
            {"start_ms": 10000, "end_ms": 14000, "speaker": "B"},
        ]
        runs = reference_solo_runs(reference, 240000)
        self.assertEqual(runs["A"], [(0, 2000), (6000, 10000)])
        self.assertEqual(runs["B"], [(10000, 14000)])
        self.assertNotIn("C", runs)
        windows = select_solo_windows(runs)
        self.assertTrue(all(
            start >= 6200 and end <= 9800
            for start, end in windows["A"]
        ))
        self.assertTrue(all(
            start >= 10200 and end <= 13800
            for start, end in windows["B"]
        ))

    def test_fourth_voice_can_yield_two_8s_windows_and_one_2s(self):
        runs = {
            "fourth": [(127300, 135400), (136200, 138600)],
            "third": [(63600, 66600)],
        }
        windows = select_solo_windows(runs)
        self.assertEqual(len(windows["fourth"]), 3)
        self.assertEqual(windows["fourth"][:2], [
            (127500, 130700), (132000, 135200),
        ])
        self.assertEqual(windows["fourth"][2], (136400, 138400))
        self.assertEqual(windows["third"], [(63800, 66400)])
        self.assertTrue(all(e - s >= 2000 for vals in windows.values()
                            for s, e in vals))

    def test_oracle_embeds_only_solo_spans_and_writes_no_vectors(self):
        wav = np.zeros(240 * 16000, dtype=np.float32)
        wav[0:3 * 16000] = 0.1
        wav[5 * 16000:13 * 16000] = 0.2
        wav[14 * 16000:17 * 16000] = 0.3
        reference = [
            {"start_ms": 0, "end_ms": 3000, "speaker": "A"},
            {"start_ms": 5000, "end_ms": 13000, "speaker": "B"},
            {"start_ms": 14000, "end_ms": 17000, "speaker": "C"},
        ]
        report = analyze_solo(reference, wav, FakeEmbedding())
        self.assertEqual(report["usable_reference_speakers"], 3)
        samples = {
            row["reference_speaker"]: row["successful_voiceprints"]
            for row in report["speakers"]
        }
        self.assertEqual(samples, {"A": 1, "B": 2, "C": 1})
        self.assertEqual(len(report["different_speaker_pair_cosines"]), 3)
        self.assertTrue(all(
            x["maximum_cosine"] == 0.0
            for x in report["different_speaker_pair_cosines"]
        ))
        self.assertNotIn("embedding", json.dumps(report))
        self.assertNotIn('"text":', json.dumps(report))

    def test_empty_or_invalid_windows_cannot_create_false_speakers(self):
        invalid = np.zeros(240 * 16000, dtype=np.float32)
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 10000, "end_ms": 13000, "speaker": "B"},
        ]
        report = analyze_solo(reference, invalid, FakeEmbedding())
        self.assertEqual(report["usable_reference_speakers"], 0)
        self.assertEqual(report["speakers"][0]["successful_voiceprints"], 0)
        self.assertEqual(report["speakers"][1]["successful_voiceprints"], 0)
        self.assertEqual(len(report["failed_windows"]), 1)
        with self.assertRaisesRegex(ValueError, "240s"):
            analyze_solo(reference, np.zeros(100), FakeEmbedding())
        with self.assertRaises(ValueError):
            reference_solo_runs(
                [{"start_ms": -1, "end_ms": 1000, "speaker": "A"}],
                240000,
            )

    def test_cosine_scores_reject_malformed_voiceprints(self):
        with self.assertRaisesRegex(ValueError, "Invalid standalone"):
            scalar_similarities(
                {"A": [np.zeros(64, dtype=np.float32)]},
                {"A": [(0, 2500)]},
            )


if __name__ == "__main__":
    unittest.main()

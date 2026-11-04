"""Reference-only clustering diagnostics: accurate pair scores and no vector leaks."""
from __future__ import annotations

import json
import unittest

import numpy as np

from tools.ami_oracle_cluster_eval import (
    evaluate_oracle_clusters, score_reference_partition,
)
from tools.ami_oracle_solo_probe import analyze_solo


def clean_probes(speaker_count=4, samples=4):
    rng = np.random.default_rng(17)
    groups = {}
    for index in range(speaker_count):
        group = []
        for _ in range(samples):
            vector = np.zeros(64, dtype=np.float32)
            vector[index] = 1.0
            vector += rng.normal(0.0, 0.015, size=64).astype(np.float32)
            vector /= np.linalg.norm(vector)
            group.append(vector)
        groups[f"AMI_{index}"] = group
    return groups


class PairwiseMetricsTests(unittest.TestCase):
    def test_permutation_invariant_precision_recall_and_adjusted_rand(self):
        truth = ["A", "A", "B", "B", "C"]
        predicted = np.array([100, 100, 100, 101, 102])
        stats = score_reference_partition(truth, predicted)
        self.assertEqual(stats["predicted_cluster_count"], 3)
        self.assertEqual(stats["true_positive_pairs"], 1)
        self.assertEqual(stats["false_positive_pairs"], 2)
        self.assertEqual(stats["false_negative_pairs"], 1)
        self.assertAlmostEqual(stats["pair_precision"], 1 / 3, places=4)
        self.assertAlmostEqual(stats["pair_recall"], 0.5, places=4)
        self.assertAlmostEqual(stats["pair_f1"], 0.4, places=5)
        self.assertAlmostEqual(stats["adjusted_rand_index"], 0.21053, places=5)
        self.assertFalse(stats["per_reference_speaker"][2]["has_repeated_samples"])

    def test_four_clean_speakers_can_be_recovered_without_oracle_k(self):
        data = clean_probes()
        report = evaluate_oracle_clusters(data)
        self.assertTrue(report["available"])
        self.assertEqual(report["successful_voiceprints"], 16)
        self.assertEqual(report["reference_speaker_count_for_evaluation_only"], 4)
        complete = [row for row in report["methods"]
                    if row["method"] == "complete_link" and row["available"]]
        self.assertTrue(complete)
        self.assertTrue(any(
            row["score"]["predicted_cluster_count"] == 4
            and row["score"]["pair_f1"] == 1.0
            for row in complete
        ))
        sphere = next(
            row for row in report["methods"]
            if row["method"] == "spherical_auto"
        )
        self.assertTrue(sphere["available"])
        self.assertEqual(sphere["score"]["predicted_cluster_count"], 4)
        self.assertEqual(sphere["score"]["pair_f1"], 1.0)
        serialized = json.dumps(report)
        self.assertNotIn('"embedding":', serialized)
        self.assertNotIn('"text":', serialized)
        self.assertNotIn('"audio":', serialized)

    def test_one_sample_speaker_is_not_claimed_to_have_intraperson_evidence(self):
        data = clean_probes(speaker_count=3, samples=4)
        data["AMI_SINGLE"] = [np.eye(64, dtype=np.float32)[40]]
        report = evaluate_oracle_clusters(data)
        self.assertTrue(report["available"])
        for method in report["methods"]:
            if method["available"]:
                single = next(
                    row for row in method["score"]["per_reference_speaker"]
                    if row["reference_speaker"] == "AMI_SINGLE"
                )
                self.assertEqual(single["sample_count"], 1)
                self.assertFalse(single["has_repeated_samples"])

    def test_insufficient_samples_and_invalid_reference_fail_closed(self):
        report = evaluate_oracle_clusters({
            "A": [np.eye(64, dtype=np.float32)[0]],
            "B": [np.eye(64, dtype=np.float32)[1]],
        })
        self.assertFalse(report["available"])
        self.assertEqual(report["successful_voiceprints"], 2)
        with self.assertRaisesRegex(ValueError, "CAM"):
            evaluate_oracle_clusters({
                "A": [np.zeros(64, dtype=np.float32)] * 4,
            })
        with self.assertRaises(ValueError):
            score_reference_partition(["A"], np.array([1]))

    def test_old_solo_probe_api_unchanged_by_default(self):
        class Fake:
            def embed(self, samples):
                v = np.zeros(64, dtype=np.float32)
                v[0] = 1
                return v

        wave = np.zeros(240 * 16000, dtype=np.float32)
        reference = [{"start_ms": 0, "end_ms": 6000, "speaker": "A"}]
        previous = analyze_solo(reference, wave, Fake())
        self.assertNotIn("clustering_evaluation", previous)
        enabled = analyze_solo(
            reference, wave, Fake(), cluster_analysis=True
        )
        self.assertFalse(enabled["clustering_evaluation"]["available"])
        self.assertEqual(
            enabled["clustering_evaluation"]["successful_voiceprints"], 2
        )


if __name__ == "__main__":
    unittest.main()

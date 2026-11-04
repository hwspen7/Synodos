"""No-model regression tests for real-world merged multi-sample speaker groups."""
from __future__ import annotations

import json
import unittest

import numpy as np

from tools.ami_cluster_substructure import explore_speaker_substructure
from tools.ami_oracle_cluster_eval import score_reference_partition


def synthetic_vectors(group_sizes=(8, 6, 12), dim=64):
    rng = np.random.default_rng(92026)
    groups = []
    for i, count in enumerate(group_sizes):
        basis = np.zeros(dim, dtype=np.float32)
        basis[i] = 1.0
        for _ in range(count):
            sample = basis + rng.normal(0, 0.01, dim).astype(np.float32)
            sample /= np.linalg.norm(sample)
            groups.append(sample)
    return np.stack(groups)


class SubstructureTests(unittest.TestCase):
    def test_mixed_large_group_can_be_reviewed_as_two_supported_groups(self):
        samples = synthetic_vectors((8, 6, 12))
        # Audio-derived first stage wrongly merged A(8) and B(6).
        baseline = np.array([0] * 14 + [1] * 12)
        truth = ["A"] * 8 + ["B"] * 6 + ["C"] * 12
        investigation = explore_speaker_substructure(samples, baseline)
        self.assertEqual(investigation.details["accepted_subdivision_count"], 1)
        self.assertEqual(investigation.details["source_cluster_count"], 2)
        self.assertIn("split_parent_0", investigation.variants)
        self.assertNotIn("split_parent_1", investigation.variants)
        before = score_reference_partition(truth, baseline)
        after = score_reference_partition(
            truth, investigation.variants["split_parent_0"],
        )
        self.assertLess(before["pair_f1"], 1.0)
        self.assertEqual(after["pair_f1"], 1.0)
        self.assertEqual(after["predicted_cluster_count"], 3)
        self.assertEqual(len(set(investigation.variants["original"])), 2)
        self.assertNotIn('"embedding":', json.dumps(investigation.details))
        self.assertNotIn('"text":', json.dumps(investigation.details))

    def test_homogeneous_groups_not_split_solely_by_large_size(self):
        samples = synthetic_vectors((20, 20))
        baseline = np.array([0] * 20 + [1] * 20)
        result = explore_speaker_substructure(samples, baseline)
        # Loose vocal variation must not make an automatic promoted speaker.
        # Every hypothetical result still remains review-only.
        self.assertEqual(
            set(result.variants).intersection({"original"}), {"original"},
        )
        self.assertFalse(any(
            "speaker_id" in item for item in result.details["proposed_groups"]
        ))
        self.assertEqual(
            result.details["source_cluster_count"], 2,
        )

    def test_small_subclusters_do_not_pass_four_sample_support(self):
        samples = synthetic_vectors((18, 2))
        baseline = np.zeros(20, dtype=np.int32)
        result = explore_speaker_substructure(samples, baseline)
        self.assertEqual(result.details["accepted_subdivision_count"], 0)
        self.assertEqual(result.details["proposed_groups"][0]["reason"],
                         "insufficient_subgroup_support")

    def test_three_coherent_source_speakers_without_reference_selection(self):
        samples = synthetic_vectors((7, 7, 7, 10))
        baseline = np.array([0] * 14 + [1] * 7 + [2] * 10)
        result = explore_speaker_substructure(samples, baseline)
        self.assertIn("split_parent_0", result.variants)
        self.assertEqual(
            len(set(result.variants["split_parent_0"])), 4,
        )

    def test_invalid_embedding_and_metadata_fail_closed(self):
        with self.assertRaises(ValueError):
            explore_speaker_substructure(
                np.zeros((4, 64), dtype=np.float32),
                np.zeros(4, dtype=np.int32),
            )
        with self.assertRaises(ValueError):
            explore_speaker_substructure(
                synthetic_vectors((4, 4)), np.zeros(8),
                min_child_samples=1,
            )
        with self.assertRaises(ValueError):
            explore_speaker_substructure(
                synthetic_vectors((4, 4)),
                np.array([0, 0, 0, 0, -1, -1, -1, -1]),
            )


if __name__ == "__main__":
    unittest.main()

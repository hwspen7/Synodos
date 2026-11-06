"""Reference-blind nested speaker split, then review-only singleton detection."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from speech.diarization.spherical_cluster import SphericalCandidate
from tools.evaluation.ami.ami_continuous_voice_audit import audit_continuous_windows


def fixture(noisy_same_person: bool = False):
    """Two raw clusters wrongly merge two stable voices plus one outlier."""
    rng = np.random.default_rng(481)
    waves = []
    reference = []
    events = []
    labels = []
    groups = (
        [("A", 8, 0)] + [("B", 6, 1)]
        + [("B" if noisy_same_person else "C", 1, 3)]
        + [("D", 12, 2)]
    )
    for name, size, coordinate in groups:
        for _ in range(size):
            vec = np.zeros(64, dtype=np.float32)
            if coordinate == 3:
                vec[1] = 0.31
                vec[3] = 0.95074
            else:
                vec[coordinate] = 1.0
                vec += rng.normal(0, 0.012, size=64).astype(np.float32)
            vec /= np.linalg.norm(vec)
            waves.append(vec)
            idx = len(waves) - 1
            start = idx * 3000
            events.append({
                "start_ms": start,
                "end_ms": start + 2000,
                "status": "embedded",
                "vector_index": idx,
            })
            reference.append({
                "start_ms": start,
                "end_ms": start + 2000,
                "speaker": name,
            })
            labels.append(0 if coordinate in (0, 1, 3) else 1)
    return waves, events, reference, np.asarray(labels, dtype=np.int32)


class HierarchicalProvisionalTests(unittest.TestCase):
    def check(self, noisy_same_person):
        embeddings, events, ref, baseline = fixture(noisy_same_person)
        initial = SphericalCandidate(baseline, {
            "method": "controlled_first_stage",
            "available": True,
            "chosen_cluster_count": 2,
        })
        with patch(
            "tools.evaluation.ami.ami_continuous_voice_audit.spherical_speaker_candidates",
            return_value=initial,
        ):
            output = audit_continuous_windows(embeddings, events, ref)
        self.assertTrue(output["available"])
        self.assertEqual(output["raw_cluster_count"], 2)
        baseline_row = next(
            item for item in output["substructure_pure_window_scores"]
            if item["variant"] == "original"
        )
        split_row = next(
            item for item in output["substructure_pure_window_scores"]
            if item["variant"] == "split_parent_0"
        )
        review = next(
            item for item in output["hierarchical_provisional_review"]
            if item["source_variant"] == "split_parent_0"
        )
        self.assertEqual(baseline_row["all_window_cluster_count"], 2)
        self.assertEqual(split_row["all_window_cluster_count"], 3)
        self.assertEqual(review["original_cluster_count"], 3)
        self.assertEqual(review["provisional_cluster_count"], 4)
        self.assertEqual(review["unconfirmed_candidate_count"], 1)
        self.assertEqual(
            review["unconfirmed_candidates"][0]["window_start_ms"],
            14 * 3000,
        )
        self.assertNotIn('"embedding":', json.dumps(review))
        self.assertNotIn('"text":', json.dumps(review))
        self.assertNotIn('"audio":', json.dumps(review))
        return split_row, review

    def test_two_stage_can_recover_rare_fourth_group_on_controlled_vectors(self):
        split, review = self.check(False)
        self.assertEqual(
            split["pure_window_score"]["false_positive_pairs"], 6
        )
        self.assertEqual(
            review["pure_window_score"]["false_positive_pairs"], 0
        )
        self.assertEqual(
            review["pure_window_score"]["false_negative_pairs"], 0
        )
        self.assertEqual(review["pure_window_score"]["pair_f1"], 1.0)

    def test_same_speaker_outlier_can_falsely_create_a_fourth_group(self):
        # Exactly the same acoustic signals, but the outlier actually
        # belongs to B. This can never be resolved by a single sample.
        split, review = self.check(True)
        self.assertEqual(
            split["pure_window_score"]["false_positive_pairs"], 0
        )
        self.assertGreater(
            review["pure_window_score"]["false_negative_pairs"], 0
        )
        self.assertLess(review["pure_window_score"]["pair_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

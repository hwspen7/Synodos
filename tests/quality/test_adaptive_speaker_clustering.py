"""Speaker-count and early-cluster regression tests without FunASR downloads."""
from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from inference.models import ModelManager
from speech.adaptive_cluster import complete_link_clusters


def voices(speakers: int, samples_per_speaker: int = 2) -> np.ndarray:
    # Deterministic synthetic acoustics, not a real-world accuracy benchmark.
    rows = []
    for i in range(speakers):
        for j in range(samples_per_speaker):
            vec = np.zeros(192, dtype=np.float32)
            vec[i] = 1.0
            vec[16 + i] = 0.025 * (j + 1)
            rows.append(vec)
    return np.asarray(rows)


class ClusterTests(unittest.TestCase):
    def test_one_to_six_speakers_before_funasr_twenty_vector_floor(self):
        for n in (1, 2, 3, 4, 6):
            with self.subTest(speakers=n):
                labels = complete_link_clusters(voices(n), threshold=0.76)
                self.assertEqual(len(set(labels)), n)
                self.assertEqual(len(labels), 2 * n)
                self.assertTrue(all(
                    labels[2 * i] == labels[2 * i + 1] for i in range(n)
                ))

    def test_no_two_speaker_limit_under_eighteen_vectors(self):
        samples = voices(6, 3)  # 18 independent voice samples
        labels = complete_link_clusters(samples, threshold=0.76)
        self.assertEqual(len(set(labels)), 6)

    def test_prevents_chain_merging_disagreeing_voices(self):
        # A ~ B and B ~ C are compatible but A and C are not:
        # complete link must not collapse the three into one cluster.
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.75, 0.66], dtype=np.float32)
        c = np.array([0.2, 0.98], dtype=np.float32)
        labels = complete_link_clusters(np.stack([a, b, c]), threshold=0.70)
        self.assertGreaterEqual(len(set(labels)), 2)

    def test_nonfinite_and_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            complete_link_clusters(np.array([[np.nan, 1.0]]))
        with self.assertRaises(ValueError):
            complete_link_clusters(voices(1), threshold=1.5)
        with self.assertRaises(ValueError):
            complete_link_clusters(voices(1), max_clusters=0)

    def test_model_manager_bootstrap_skips_legacy_cluster(self):
        model = ModelManager.__new__(ModelManager)
        model._infer_lock = threading.RLock()
        calls = []
        model._load_cluster = lambda: lambda embs, oracle_num=None: (
            calls.append(len(embs)) or np.zeros(len(embs), dtype=int)
        )
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "bootstrap",
        }):
            labels = model._cluster_embeddings(voices(4))
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(calls, [])
            labels_large = model._cluster_embeddings(voices(6, 4))
            self.assertEqual(len(set(labels_large)), 1)
            self.assertEqual(calls, [24])

    def test_candidate_opt_in_reclusters_24_samples_without_two_person_cap(self):
        model = ModelManager.__new__(ModelManager)
        model._infer_lock = threading.RLock()
        model._load_cluster = lambda: lambda embs, oracle_num=None: (
            np.zeros(len(embs), dtype=int)
        )
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "candidate",
        }):
            labels = model._cluster_embeddings(voices(6, 4))
        self.assertEqual(len(set(labels)), 6)

    def test_legacy_mode_preserves_previous_behavior(self):
        model = ModelManager.__new__(ModelManager)
        model._infer_lock = threading.RLock()
        model._load_cluster = lambda: lambda embs, oracle_num=None: (
            np.zeros(len(embs), dtype=int)
        )
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "legacy",
        }):
            self.assertEqual(len(set(model._cluster_embeddings(voices(4)))), 1)


if __name__ == "__main__":
    unittest.main()

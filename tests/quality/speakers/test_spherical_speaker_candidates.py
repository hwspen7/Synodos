"""Regression tests for non-fragmenting candidate multi-speaker selection."""
from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from inference.models import ModelManager
from speech.diarization.adaptive_cluster import complete_link_clusters
from speech.diarization.spherical_cluster import spherical_speaker_candidates


def noisy_voices(k=4, each=22, dim=128):
    """Synthetic weak within-speaker cosines, not an AMI score or tuning data."""
    rng = np.random.default_rng(2026)
    rows = []
    for speaker in range(k):
        prototype = np.zeros(dim)
        prototype[speaker] = 0.55
        for _ in range(each):
            rows.append(prototype + 0.05 * rng.normal(size=dim))
    return np.asarray(rows, dtype=np.float32)


def mock_model():
    model = ModelManager.__new__(ModelManager)
    model._infer_lock = threading.RLock()
    model._load_cluster = lambda: lambda data, oracle_num=None: np.zeros(
        len(data), dtype=np.int32
    )
    return model


class SphericalCandidateTests(unittest.TestCase):
    def test_noisy_88_samples_use_spherical_clusters_not_32_strict_clusters(self):
        samples = noisy_voices()
        with self.assertRaises(ValueError):
            complete_link_clusters(samples, threshold=0.76)
        result = spherical_speaker_candidates(samples)
        self.assertTrue(result.details["available"])
        self.assertIsNotNone(result.labels)
        self.assertEqual(result.details["chosen_cluster_count"], 4)
        self.assertEqual(len(set(result.labels)), 4)
        self.assertGreaterEqual(result.details["smallest_cluster"], 2)

    def test_not_hardcoded_four_speakers(self):
        for k in (2, 3, 4, 6):
            with self.subTest(speakers=k):
                result = spherical_speaker_candidates(
                    noisy_voices(k=k, each=10),
                )
                self.assertTrue(result.details["available"])
                self.assertEqual(len(set(result.labels)), k)

    def test_unknown_random_voiceprints_abstain_or_report_nonoracle(self):
        rng = np.random.default_rng(78)
        result = spherical_speaker_candidates(
            rng.normal(size=(64, 256)).astype(np.float32)
        )
        if result.labels is None:
            self.assertFalse(result.details["available"])
        else:
            self.assertTrue(result.details["available"])
            self.assertIn("warning", result.details)
            self.assertTrue(2 <= result.details["chosen_cluster_count"] <= 8)

    def test_candidate_execution_receipt_cannot_silently_use_funasr(self):
        model = mock_model()
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "candidate",
        }):
            labels, receipt = model._cluster_embeddings_with_info(noisy_voices())
        self.assertEqual(receipt["strategy_requested"], "candidate")
        self.assertEqual(receipt["strategy_executed"], "spherical_candidate")
        self.assertIsNone(receipt["fallback_reason"])
        self.assertEqual(len(set(labels)), 4)
        self.assertEqual(receipt["cluster_count"], 4)
        self.assertNotIn("embeddings_raw", str(receipt))

    def test_forced_candidate_abstention_reports_funasr_fallback(self):
        model = mock_model()
        rng = np.random.default_rng(167)
        samples = rng.normal(size=(64, 256)).astype(np.float32)
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "candidate",
        }), patch(
            "speech.diarization.spherical_cluster.spherical_speaker_candidates",
        ) as selected:
            from speech.diarization.spherical_cluster import SphericalCandidate
            selected.return_value = SphericalCandidate(None, {
                "available": False, "reason": "weak_acoustic_separation",
            })
            labels, receipt = model._cluster_embeddings_with_info(samples)
        self.assertEqual(receipt["strategy_executed"], "funasr_legacy")
        self.assertEqual(receipt["fallback_reason"], "weak_acoustic_separation")
        self.assertEqual(len(set(labels)), 1)

    def test_bootstrap_and_legacy_paths_still_supported(self):
        model = mock_model()
        from tests.quality.speakers.test_adaptive_speaker_clustering import voices
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "bootstrap"
        }):
            labels, receipt = model._cluster_embeddings_with_info(voices(4))
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(receipt["strategy_executed"], "complete_link")
            _, receipt = model._cluster_embeddings_with_info(voices(4, 5))
            self.assertEqual(receipt["strategy_executed"], "funasr_legacy")
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "legacy"
        }):
            _, receipt = model._cluster_embeddings_with_info(voices(4))
            self.assertEqual(receipt["strategy_executed"], "funasr_legacy")


if __name__ == "__main__":
    unittest.main()

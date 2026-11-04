"""Speaker quality candidate stays independent and reversible.

No unit test can establish a lower diarization error rate.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.models import ModelManager
from sessions.processor import SessionProcessor
from speech.quality_profile import speaker_candidate_enabled


class SpeakerCandidateTests(unittest.TestCase):
    def test_flag_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(speaker_candidate_enabled())
        with patch.dict(os.environ, {"MEETING_SPEAKER_QUALITY_CANDIDATE": "1"}):
            self.assertTrue(speaker_candidate_enabled())

    def test_strategy_is_independent_and_legacy_is_explicitly_respected(self):
        model = object.__new__(ModelManager)
        samples = np.array([
            [1.0, 0.0], [0.99, 0.10], [0.0, 1.0],
        ], dtype=np.float32)
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_QUALITY_CANDIDATE": "1",
        }, clear=True):
            labels, receipt = model._cluster_embeddings_with_info(samples)
            self.assertEqual(len(labels), 3)
            self.assertEqual(receipt["strategy_requested"], "candidate")
            self.assertEqual(receipt["strategy_executed"], "complete_link")
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_QUALITY_CANDIDATE": "1",
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "bootstrap",
        }, clear=True):
            _, receipt = model._cluster_embeddings_with_info(samples)
            self.assertEqual(receipt["strategy_requested"], "bootstrap")

    def _finalizer(self):
        processor = object.__new__(SessionProcessor)
        processor.buffer = SimpleNamespace(total_samples=lambda: 0)
        processor._collect_speakers = Mock(return_value=False)
        processor._collect_acoustic = Mock(return_value=False)
        processor._recluster = Mock()
        processor._fallback_labels = Mock()
        processor._collect_translations = Mock()
        processor._rebuild_and_push = Mock()
        processor._do_save = Mock(return_value=True)
        processor._put = Mock()
        processor._save_queue = SimpleNamespace(close=Mock())
        processor._next_segment_id = 3
        processor._speaker_tasks = {}
        processor._acoustic_tasks = {}
        processor._translation_tasks = {}
        processor._final_save_ok = False
        processor.meeting_store = None
        processor.meeting_id = None
        return processor

    def test_final_candidate_rechecks_clusters_even_without_new_tasks(self):
        processor = self._finalizer()
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_QUALITY_CANDIDATE": "1",
        }, clear=True):
            processor._finalize()
        processor._recluster.assert_called_once_with(
            allow_historical_relabel=True,
        )
        self.assertEqual(processor._last_reclustered_segment_id, 2)

    def test_default_finalization_does_not_force_reclustering(self):
        processor = self._finalizer()
        with patch.dict(os.environ, {}, clear=True):
            processor._finalize()
        processor._recluster.assert_not_called()

    def test_explicit_reclustering_can_bypass_mature_label_lock(self):
        processor = object.__new__(SessionProcessor)
        processor.segments = [{
            "id": 1,
            "start_sample": 0, "end_sample": 32000,
            "text": "Hello", "emb": np.array([1., 0.]),
            "cluster": -1, "speaker_id": 8, "speaker_name": "Speaker 8",
        }]
        processor.settings = SimpleNamespace(sample_rate=16000)
        processor._match_participant = Mock(return_value=(None, 0.0))
        processor.models = SimpleNamespace(
            cluster_with_info=lambda embeddings: (
                np.array([0]), {"strategy_executed": "test"},
            ),
        )
        processor._remap_speakers = Mock()
        processor._assign_labels = Mock()
        with patch.dict(os.environ, {}, clear=True), patch(
            "sessions.processor.collect_mature_labels",
            side_effect=AssertionError("historical freeze must be bypassed"),
        ):
            processor._recluster(allow_historical_relabel=True)
        processor._assign_labels.assert_called_once()


if __name__ == "__main__":
    unittest.main()

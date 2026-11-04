"""Speaker quality candidate stays independent and reversible.

No unit test can establish a lower diarization error rate.
"""
from __future__ import annotations

import os
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from inference.models import ModelManager
from sessions.processor import SessionProcessor
from speech.quality_profile import speaker_candidate_enabled


class SpeakerCandidateTests(unittest.TestCase):
    def test_flag_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(speaker_candidate_enabled())
        with patch.dict(os.environ, {"MEETING_SPEAKER_QUALITY_CANDIDATE": "1"}):
            self.assertTrue(speaker_candidate_enabled())

    def test_probe_floor_changes_only_for_speaker_candidate(self):
        # This unit test exercises speaker probe policy, not ModelScope
        # discovery. Explicit model IDs avoid Path.home() when the test
        # deliberately clears HOME/USERPROFILE from its environment.
        settings = Settings(
            asr_model="test-asr",
            fsmn_vad_model="test-vad",
            punc_model="test-punc",
            spk_model="test-speaker",
        )
        with patch.dict(os.environ, {}, clear=True):
            reference = SessionProcessor(
                "reference", AudioBuffer(16000, max_seconds=30),
                settings, object(), asyncio.Queue(),
            )
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_QUALITY_CANDIDATE": "1",
        }, clear=True):
            candidate = SessionProcessor(
                "candidate", AudioBuffer(16000, max_seconds=30),
                settings, object(), asyncio.Queue(),
            )
        self.assertEqual(reference._speaker_min_audio_s, 1.2)
        self.assertEqual(candidate._speaker_min_audio_s, 0.9)

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

    def test_short_acoustic_handoff_remains_opt_in(self):
        from speech.turn_split import split_by_acoustic_turns

        audio = np.concatenate([
            np.full(20000, 0.1, dtype=np.float32),
            np.full(20000, 0.2, dtype=np.float32),
        ])
        source = {
            "start": 0, "end": 2500, "text": "Hello world",
            "words": [
                {"start_ms": 0, "end_ms": 1250, "text": "Hello"},
                {"start_ms": 1250, "end_ms": 2500, "text": "world"},
            ],
        }

        def embed(window):
            return np.array(
                [1.0, 0.0] if float(np.mean(window)) < 0.15
                else [0.5, np.sqrt(0.75)],
            )

        with patch.dict(os.environ, {}, clear=True):
            reference = split_by_acoustic_turns(source, audio, embed)
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_QUALITY_CANDIDATE": "1",
        }, clear=True):
            candidate = split_by_acoustic_turns(source, audio, embed)

        self.assertEqual(len(reference), 1)
        self.assertEqual(len(candidate), 2)
        self.assertEqual([part["text"] for part in candidate], [
            "Hello", "world",
        ])

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

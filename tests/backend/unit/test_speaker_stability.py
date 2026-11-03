"""Streaming speaker identity regression and long-meeting label stability."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.speaker_stability import (
    collect_mature_labels, restore_mature_labels, stabilized_cluster_ids,
)
from speech.config import SpeechSettings


class StabilityUnitContracts(unittest.TestCase):
    def test_raw_cluster_renumbering_keeps_historical_speaker_ids(self):
        items = [
            {"id": i, "cluster": 31, "speaker_id": 2}
            for i in range(1, 61)
        ] + [
            {"id": i, "cluster": 7, "speaker_id": 1}
            for i in range(61, 161)
        ]
        assignment = stabilized_cluster_ids(items, iter([]), {7: 2, 31: 1})
        self.assertEqual(assignment, {7: 1, 31: 2})
        # A second raw renumbering must not create new speaker identities.
        for item in items:
            item["cluster"] = 5 if item["speaker_id"] == 1 else 42
        result = stabilized_cluster_ids(items, (), assignment)
        self.assertEqual(result[5], 1)
        self.assertEqual(result[42], 2)

    def test_large_continuing_cluster_beats_small_split_off(self):
        # Old sorted-raw greedy mapping stole stable 1 for the two-clip
        # split-off (raw 0) and rewrote 100 continuing clips (raw 1).
        items = [
            {"id": i, "cluster": 0, "speaker_id": 1}
            for i in range(2)
        ] + [
            {"id": i + 10, "cluster": 1, "speaker_id": 1}
            for i in range(100)
        ]
        result = stabilized_cluster_ids(items, (), {0: 1})
        self.assertEqual(result[1], 1)
        self.assertNotEqual(result[0], 1)
        self.assertGreater(result[0], 1)

    def test_participant_ids_are_reserved_and_never_recycled(self):
        items = [
            {"id": 1, "cluster": 0, "speaker_id": 3},
            {"id": 2, "cluster": 2, "speaker_id": -2},
        ]
        result = stabilized_cluster_ids(items, iter([1, 2]), {2: 4})
        self.assertEqual(result[0], 3)
        self.assertEqual(result[2], 4)
        self.assertEqual(len(set(result.values())), len(result))
        new = stabilized_cluster_ids(items + [
            {"id": 3, "cluster": 7, "speaker_id": -3},
        ], (1, 2), {0: 3, 2: 4})
        self.assertGreater(new[7], 4)

    def test_mature_anchors_preserve_only_confident_unedited_labels(self):
        rate = 16000
        records = [
            {"id": 1, "start_sample": 0, "end_sample": 10000,
             "speaker_id": 1, "speaker_name": "Speaker 1",
             "emb": np.array([1, 0]), "speaker_uncertain": False},
            {"id": 2, "start_sample": 0, "end_sample": 20000,
             "speaker_id": -2, "speaker_name": "Unidentified Speaker",
             "emb": None},
            {"id": 3, "start_sample": 0, "end_sample": 30000,
             "speaker_id": 3, "speaker_name": "Speaker 3",
             "speaker_uncertain": True, "emb": np.array([0, 1])},
            {"id": 4, "start_sample": 0, "end_sample": 40000,
             "speaker_id": 4, "speaker_name": "Speaker 4",
             "emb": np.array([1, 0]), "speaker_override": True},
            {"id": 5, "start_sample": 0, "end_sample": 200 * rate,
             "speaker_id": 5, "speaker_name": "Speaker 5",
             "emb": np.array([0, 1])},
        ]
        locked = collect_mature_labels(records, rate, 120)
        self.assertEqual(locked, {1: (1, "Speaker 1")})
        for r in records:
            r["speaker_id"] = 9
            r["speaker_name"] = "Speaker 9"
        self.assertEqual(restore_mature_labels(records, locked), 1)
        self.assertEqual(records[0]["speaker_id"], 1)
        self.assertEqual(records[1]["speaker_id"], 9)
        self.assertEqual(collect_mature_labels(records, rate, 0), {})
        with self.assertRaises(ValueError):
            collect_mature_labels(records, rate, 4000)


class IntegrationSpeakerStabilityContracts(unittest.TestCase):
    def make_processor(self):
        cfg = Settings()
        class Model:
            def cluster(self, embeddings):
                return np.full((len(embeddings),), 8, dtype=np.int32)
        with patch("sessions.processor.load_settings", return_value=SpeechSettings(
                engine="whisper", model="large-v3")):
            proc = SessionProcessor("speaker-stability", AudioBuffer(16000),
                                    cfg, Model(), asyncio.Queue())
        proc._speaker_relabel_horizon_s = 120
        return proc

    def test_long_session_preserves_old_trusted_identity_but_new_moves(self):
        proc = self.make_processor()
        sr = proc.settings.sample_rate
        proc.segments = [
            {"id": 1, "start_sample": 0, "end_sample": sr,
             "speaker_id": 1, "speaker_name": "Speaker 1", "cluster": 1,
             "speaker_uncertain": False, "emb": np.array([1., 0.])},
        ]
        proc.segments.extend({
            "id": i, "start_sample": (90 + i) * sr,
            "end_sample": (91 + i) * sr,
            "speaker_id": 2, "speaker_name": "Speaker 2", "cluster": 2,
            "speaker_uncertain": False, "emb": np.array([0., 1.]),
        } for i in range(2, 80))
        # All new cluster labels become 8. Speaker 2 wins by majority.
        proc._recluster()
        self.assertEqual(proc.segments[0]["speaker_id"], 1)
        self.assertEqual(proc.segments[-1]["speaker_id"], 2)
        self.assertEqual(proc._speaker_map[8], 2)

        # Legacy behavior is intentionally available as an A/B control.
        proc._speaker_relabel_horizon_s = 0
        proc._recluster()
        self.assertEqual(proc.segments[0]["speaker_id"], 2)

    def test_invalid_env_horizon_rejected(self):
        with patch.dict(os.environ, {"MEETING_SPEAKER_LABEL_REVISABLE_S": "4000"}):
            with self.assertRaisesRegex(ValueError, "REVISABLE_S"):
                self.make_processor()


if __name__ == "__main__":
    unittest.main()

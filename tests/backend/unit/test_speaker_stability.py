"""Streaming speaker identity regression and long-meeting label stability."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from speech.diarization import (
    SpeakerCentroidTracker, assign_acoustic_anchor_clusters,
)
from sessions.processor import SessionProcessor
from sessions.speaker_stability import (
    collect_mature_labels, restore_mature_labels, stabilized_cluster_ids,
    summarize_cluster_fragmentation, diagnose_cluster_merge_thresholds,
    anonymous_speaker_timeline,
)
from speech.config import SpeechSettings


class StabilityUnitContracts(unittest.TestCase):
    def test_anonymous_speaker_timeline_is_bounded_and_text_free(self):
        records = [
            {
                "id": 4, "start_sample": 16000, "end_sample": 32000,
                "cluster": 2, "speaker_id": 7,
                "speaker_uncertain": False,
                "text": "Private meeting topic",
                "speaker_name": "Private participant",
                "emb": np.ones(192, dtype=np.float32),
            },
            {
                "id": 3, "start_sample": 0, "end_sample": 8000,
                "cluster": -3, "speaker_id": -3,
                "speaker_uncertain": True,
                "text": "Private first sentence",
                "speaker_name": "Unidentified Speaker",
                "emb": None,
            },
        ]
        report = anonymous_speaker_timeline(
            records, 16000, max_segments=1)
        self.assertEqual(report["total_segments"], 2)
        self.assertEqual(report["exported_segments"], 1)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["segments"][0]["segment_id"], 3)
        self.assertEqual(report["segments"][0]["start_ms"], 0)
        self.assertIsNone(report["segments"][0]["raw_cluster_id"])
        self.assertIsNone(report["segments"][0]["published_speaker_id"])
        full = anonymous_speaker_timeline(records, 16000)
        self.assertEqual([x["segment_id"] for x in full["segments"]],
                         [3, 4])
        self.assertEqual(full["segments"][1]["raw_cluster_id"], 2)
        self.assertEqual(full["segments"][1]["published_speaker_id"], 7)
        self.assertEqual(full["segments"][1]["start_ms"], 1000)
        self.assertEqual(full["segments"][1]["end_ms"], 2000)
        self.assertTrue(full["segments"][1]["embedding_available"])
        self.assertFalse(full["truncated"])
        serialized = str(full)
        self.assertNotIn("Private", serialized)
        self.assertNotIn("array(", serialized)
        with self.assertRaises(ValueError):
            anonymous_speaker_timeline(records, 0)
        with self.assertRaises(ValueError):
            anonymous_speaker_timeline(records, 16000,
                                       max_segments=1000)

    def test_posthoc_merge_sensitivity_separates_similar_cluster_pairs(self):
        records = [
            {"cluster": 0, "emb": np.array([1.0, 0.0])},
            {"cluster": 1, "emb": np.array([0.7, 0.714])},
            {"cluster": 2, "emb": np.array([-1.0, 0.0])},
            {"cluster": 3, "emb": np.array([-0.7, -0.714])},
        ]
        result = diagnose_cluster_merge_thresholds(
            records, (0.65, 0.78, 0.90))
        self.assertTrue(result["sweep_available"])
        self.assertEqual(result["starting_raw_cluster_count"], 4)
        self.assertEqual(
            [row["posthoc_cluster_count"]
             for row in result["candidate_merges"]], [2, 4, 4])
        self.assertNotIn("emb", str(result))

    def test_posthoc_merge_sweep_skips_unusable_embeddings(self):
        self.assertFalse(diagnose_cluster_merge_thresholds(
            [{"cluster": 2, "emb": None}]
        )["sweep_available"])
        with self.assertRaises(ValueError):
            diagnose_cluster_merge_thresholds([], (0.0,))
        result = diagnose_cluster_merge_thresholds([
            {"cluster": 0, "emb": [1.0, 0.0]},
            {"cluster": 1, "emb": [float("nan"), 1.0]},
            {"cluster": 2, "emb": [0.0, 1.0, 0.0]},
        ])
        self.assertTrue(result["sweep_available"])
        self.assertEqual(result["starting_raw_cluster_count"], 1)
        self.assertTrue(all(
            row["posthoc_cluster_count"] == 1
            for row in result["candidate_merges"]
        ))

    def test_final_raw_clusters_explain_frozen_label_fragmentation(self):
        # Two raw acoustic clusters but six frozen display labels
        # isolates identity alias mapping from an acoustic split.
        records = [
            {"id": i, "cluster": i % 2, "speaker_id": i // 2 + 1,
             "emb": np.array([1., 0.])}
            for i in range(12)
        ]
        result = summarize_cluster_fragmentation(records)
        self.assertEqual(result["raw_cluster_count"], 2)
        self.assertEqual(result["published_positive_speaker_count"], 6)
        self.assertEqual(result["raw_clusters_with_multiple_published_labels"], 2)
        self.assertEqual(result["max_published_labels_per_raw_cluster"], 6)
        self.assertEqual(result["labeled_segments_in_conflicted_raw_clusters"], 12)
        self.assertEqual(
            result["published_labels_spanning_multiple_raw_clusters"], 6)
        self.assertEqual(result["largest_raw_cluster_size"], 6)
        self.assertEqual(len(result["top_conflicted_raw_clusters"]), 2)

    def test_final_raw_clusters_can_be_true_acoustic_fragmentation(self):
        records = [
            {"id": i, "cluster": i, "speaker_id": i + 1,
             "emb": np.array([1., 0.])}
            for i in range(5)
        ]
        result = summarize_cluster_fragmentation(records)
        self.assertEqual(result["raw_cluster_count"], 5)
        self.assertEqual(result["published_positive_speaker_count"], 5)
        self.assertEqual(result["raw_clusters_with_multiple_published_labels"], 0)
        self.assertEqual(result["labeled_segments_in_conflicted_raw_clusters"], 0)
        self.assertEqual(result["top_conflicted_raw_clusters"], [])
        missing = summarize_cluster_fragmentation([
            {"cluster": -3, "speaker_id": -1, "emb": None},
        ])
        self.assertEqual(missing["raw_cluster_count"], 0)
        self.assertEqual(missing["segments_without_embedding"], 1)

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

    def test_anchor_clusters_reject_new_identity_from_short_outliers(self):
        sr = 16000
        records = [
            {"id": 1, "start_sample": 0, "end_sample": sr * 2,
             "emb": np.array([1., 0.]), "speaker_uncertain": False},
            {"id": 2, "start_sample": sr * 2, "end_sample": sr * 4,
             "emb": np.array([0., 1.]), "speaker_uncertain": False},
            {"id": 3, "start_sample": sr * 4, "end_sample": sr * 4 + 3200,
             "emb": np.array([0.99, 0.01]), "speaker_uncertain": False},
            {"id": 4, "start_sample": sr * 5, "end_sample": sr * 5 + 3200,
             "emb": np.array([0.707, 0.707]), "speaker_uncertain": False},
            {"id": 5, "start_sample": sr * 6, "end_sample": sr * 6 + 3200,
             "emb": np.array([-0.9, -0.1]), "speaker_uncertain": True},
        ]
        calls = []
        def cluster(vectors):
            calls.append(vectors.copy())
            return np.where(vectors[:, 0] > vectors[:, 1], 3, 8)
        stats = assign_acoustic_anchor_clusters(
            records, list(range(5)), cluster, sr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (2, 2))
        self.assertEqual(stats, {
            "anchor_segments": 2, "short_assigned": 1,
            "unassigned": 2, "raw_clusters": 2,
        })
        self.assertEqual(
            [r["cluster"] for r in records], [3, 8, 3, -3, -3])

    def test_anchor_mode_does_not_force_ids_without_long_evidence(self):
        records = [
            {"id": 1, "start_sample": 0, "end_sample": 3000,
             "emb": np.array([1., 0.]), "speaker_uncertain": False},
        ]
        def must_not_cluster(_):
            raise AssertionError("Short speech must not become an anchor")
        stats = assign_acoustic_anchor_clusters(
            records, [0], must_not_cluster, 16000)
        self.assertEqual(stats["raw_clusters"], 0)
        self.assertEqual(stats["unassigned"], 1)
        self.assertEqual(records[0]["cluster"], -3)
        with self.assertRaises(ValueError):
            assign_acoustic_anchor_clusters(records, [0],
                                            must_not_cluster, 16000,
                                            short_match_margin=-0.1)

    def test_centroid_seed_uses_all_existing_voice_examples(self):
        tracker = SpeakerCentroidTracker(threshold=0.72)
        tracker.seed(1, np.array([0.6, 0.8], dtype=np.float32))
        for _ in range(10):
            tracker.seed(1, np.array([1.0, 0.0], dtype=np.float32))
        tracker.seed(2, np.array([-1.0, 0.0], dtype=np.float32))
        self.assertEqual(tracker.counts[1], 11)
        self.assertGreater(tracker.centroids[1][0], 0.98)
        self.assertEqual(tracker.assign(np.array([1.0, 0.0])), 1)
        self.assertEqual(tracker.assign(np.array([-1.0, 0.0])), 2)
        self.assertEqual(len(tracker.centroids), 2)

    def test_centroid_seed_rejects_bad_or_incompatible_vectors(self):
        tracker = SpeakerCentroidTracker(threshold=0.72)
        tracker.seed(1, np.array([1.0, 0.0]))
        tracker.seed(1, np.array([float("nan"), 1.0]))
        tracker.seed(1, np.array([1.0, 0.0, 0.0]))
        tracker.seed(-1, np.array([1.0, 0.0]))
        self.assertEqual(tracker.counts, {1: 1})
        self.assertEqual(len(tracker.centroids), 1)

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

    def test_centroid_mode_reuses_historical_speaker_prototypes(self):
        proc = self.make_processor()
        sr = proc.settings.sample_rate
        vectors = [
            (1, np.array([0.6, 0.8], dtype=np.float32))
        ] + [
            (1, np.array([1.0, 0.0], dtype=np.float32))
            for _ in range(10)
        ] + [
            (2, np.array([-1.0, 0.0], dtype=np.float32)),
            (-1, np.array([1.0, 0.0], dtype=np.float32)),
            (-1, np.array([-1.0, 0.0], dtype=np.float32)),
        ]
        proc.segments = [
            {
                "id": i + 1,
                "start_sample": sr * i,
                "end_sample": sr * (i + 1),
                "speaker_id": sid,
                "speaker_name": (
                    f"Speaker {sid}" if sid > 0 else "Unidentified Speaker"
                ),
                "cluster": -2, "emb": vector,
            }
            for i, (sid, vector) in enumerate(vectors)
        ]
        with patch.dict(os.environ, {"MEETING_DIARIZATION_MODE": "centroid"}):
            proc._recluster()
        self.assertEqual(proc.segments[-2]["speaker_id"], 1)
        self.assertEqual(proc.segments[-1]["speaker_id"], 2)
        self.assertEqual({
            s["speaker_id"] for s in proc.segments
            if s["speaker_id"] > 0
        }, {1, 2})

    def test_opt_in_anchor_mode_uses_existing_clusters_without_short_seeds(self):
        proc = self.make_processor()
        sr = proc.settings.sample_rate
        class Model:
            def __init__(self):
                self.inputs = []
            def cluster(self, embeddings):
                self.inputs.append(embeddings.copy())
                return np.where(embeddings[:, 0] > embeddings[:, 1], 0, 1)
        proc.models = Model()
        proc.segments = [
            {
                "id": i, "start_sample": start, "end_sample": end,
                "speaker_id": -i, "speaker_name": "Unidentified Speaker",
                "speaker_uncertain": False, "cluster": -3,
                "emb": np.array(vector, dtype=np.float32),
            }
            for i, start, end, vector in [
                (1, 0, 2 * sr, [1., 0.]),
                (2, 2 * sr, 4 * sr, [0., 1.]),
                (3, 4 * sr, 4 * sr + 3200, [0.99, 0.01]),
                (4, 5 * sr, 5 * sr + 3200, [0.707, 0.707]),
            ]
        ]
        with patch.dict(os.environ,
                        {"MEETING_DIARIZATION_MODE": "cluster_anchors"}):
            proc._recluster()
        self.assertEqual(proc.models.inputs[0].shape, (2, 2))
        self.assertEqual(proc.segments[0]["speaker_id"],
                         proc.segments[2]["speaker_id"])
        self.assertNotEqual(proc.segments[0]["speaker_id"],
                            proc.segments[1]["speaker_id"])
        self.assertLess(proc.segments[3]["speaker_id"], 0)
        self.assertEqual(len({
            s["speaker_id"] for s in proc.segments
            if s["speaker_id"] > 0
        }), 2)

    def test_opt_in_sticky_global_clustering_preserves_reliable_labels(self):
        proc = self.make_processor()
        sr = proc.settings.sample_rate
        proc.segments = [
            {
                "id": i, "start_sample": (i - 1) * sr,
                "end_sample": i * sr,
                "speaker_id": sid,
                "speaker_name": (
                    f"Speaker {sid}" if sid > 0 else "Unidentified Speaker"
                ),
                "speaker_uncertain": False if sid > 0 else True,
                "cluster": i,
                "emb": np.array([1., 0.], dtype=np.float32)
                if sid == 1 else np.array([0., 1.], dtype=np.float32),
            }
            for i, sid in [(1, 1), (2, 2), (3, -3)]
        ]
        # A global cluster merge should not rename a previously reliable
        # speaker. New unknown utterances are still assigned acoustically.
        with patch.dict(os.environ,
                        {"MEETING_DIARIZATION_MODE": "cluster_sticky"}):
            proc._recluster()
        self.assertEqual([s["speaker_id"] for s in proc.segments[:2]],
                         [1, 2])
        self.assertGreater(proc.segments[2]["speaker_id"], 0)
        self.assertEqual(len({s["speaker_id"] for s in proc.segments[:2]}), 2)

    def test_standard_cluster_retains_existing_relabel_behavior(self):
        proc = self.make_processor()
        sr = proc.settings.sample_rate
        proc.segments = [
            {
                "id": i, "start_sample": (i - 1) * sr,
                "end_sample": i * sr,
                "speaker_id": i,
                "speaker_name": f"Speaker {i}",
                "speaker_uncertain": False, "cluster": i,
                "emb": np.array([float(i), 1.], dtype=np.float32),
            }
            for i in [1, 2]
        ]
        with patch.dict(os.environ, {"MEETING_DIARIZATION_MODE": "cluster"}):
            proc._recluster()
        # The synthetic model merges both acoustic clusters, so standard
        # mode may revise earlier IDs as before, unlike sticky comparator.
        self.assertEqual(proc.segments[0]["speaker_id"],
                         proc.segments[1]["speaker_id"])

    def test_invalid_env_horizon_rejected(self):
        with patch.dict(os.environ, {"MEETING_SPEAKER_LABEL_REVISABLE_S": "4000"}):
            with self.assertRaisesRegex(ValueError, "REVISABLE_S"):
                self.make_processor()


if __name__ == "__main__":
    unittest.main()

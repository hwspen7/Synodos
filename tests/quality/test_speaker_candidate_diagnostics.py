"""Fresh acoustic clustering diagnostics must detect split hypotheses.

Tests are synthetic and do not assert real-world DER improvements.
"""
from __future__ import annotations

import json
import unittest
import numpy as np

from sessions.speaker_stability import (
    anonymous_speaker_timeline,
    diagnose_fresh_cluster_candidates,
    summarize_cluster_fragmentation,
)


class FreshClusterTests(unittest.TestCase):
    def test_recovers_six_candidate_groups_from_one_raw_cluster(self):
        segments = []
        for speaker in range(6):
            for turn in range(3):
                vec = np.zeros(192, dtype=np.float32)
                vec[speaker] = 1.0
                vec[40 + speaker] = turn * 0.03
                segments.append({
                    "id": len(segments) + 1,
                    "cluster": 0, "speaker_id": 1,
                    "start_sample": len(segments) * 16000,
                    "end_sample": (len(segments) + 1) * 16000,
                    "emb": vec, "speaker_evidence_status": "turn_voiceprint",
                    "text": "private speech text is never exported",
                })
        baseline = summarize_cluster_fragmentation(segments)
        self.assertEqual(baseline["raw_cluster_count"], 1)
        self.assertEqual(baseline["published_positive_speaker_count"], 1)
        self.assertEqual(baseline["speaker_evidence_status_counts"]["turn_voiceprint"], 18)
        fresh = diagnose_fresh_cluster_candidates(segments)
        self.assertTrue(fresh["available"])
        for row in fresh["candidate_reclustering"]:
            self.assertEqual(row["candidate_cluster_count"], 6)
            self.assertEqual(row["singleton_clusters"], 0)
        self.assertNotIn("private speech text", json.dumps(fresh))
        timeline = anonymous_speaker_timeline(segments, 16000)
        self.assertEqual(
            timeline["segments"][0]["speaker_evidence_status"],
            "turn_voiceprint",
        )
        self.assertNotIn("private speech text", json.dumps(timeline))

    def test_empty_and_too_many_vectors_refuse_candidate_counts(self):
        self.assertFalse(diagnose_fresh_cluster_candidates([])["available"])
        records = [
            {"cluster": 0, "emb": np.array([1.0, 0.0], dtype=np.float32)}
            for _ in range(257)
        ]
        result = diagnose_fresh_cluster_candidates(records)
        self.assertFalse(result["available"])
        self.assertEqual(result["eligible_embeddings"], 257)


if __name__ == "__main__":
    unittest.main()

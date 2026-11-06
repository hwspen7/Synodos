"""Prevent a single reader from becoming Speaker 1-6 in the default ASR path.

Synthetic vectors establish policy correctness, not real-world DER.
"""
from __future__ import annotations

import unittest
import numpy as np

from sessions.speaker_cluster_birth import gate_new_cluster_births


def row(number, cluster, vector, start, end, **extra):
    return {
        "id": number, "cluster": cluster, "emb": (
            np.asarray(vector, dtype=np.float32) if vector is not None else None
        ),
        "start_sample": int(start * 16000),
        "end_sample": int(end * 16000),
        "text": f"Utterance {number}", "speaker_uncertain": False,
        "speaker_evidence_status": "turn_voiceprint",
        "timestamp_quality": "segment", "words": [],
        **extra,
    }


class SpeakerClusterBirthTests(unittest.TestCase):
    def test_one_reader_split_into_six_raw_clusters_reuses_first_person(self):
        # A changing speaking style can make complete-link clusters differ.
        # Verified matching voices must not allocate five new display IDs.
        vectors = [
            [1.0, 0], [.99, .08], [.98, -.10],
            [.96, .19], [.97, -.12], [.99, .06],
        ]
        turns = [row(i + 1, i, vector, i * 4, i * 4 + 2.7)
                 for i, vector in enumerate(vectors)]
        preserved = [
            (s["id"], s["text"], s["start_sample"], s["end_sample"])
            for s in turns
        ]
        stats = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(stats, {"reused": 5, "deferred": 0, "confirmed": 1})
        self.assertEqual([s["cluster"] for s in turns], [0] * 6)
        self.assertEqual(
            [(s["id"], s["text"], s["start_sample"], s["end_sample"])
             for s in turns], preserved,
        )

    def test_noncorroborated_short_new_speaker_cannot_get_new_id(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [0, 1], 3, 5),
        ]
        stats = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(stats, {"reused": 0, "deferred": 1, "confirmed": 1})
        self.assertEqual([s["cluster"] for s in turns], [0, -3])
        self.assertEqual(turns[1]["speaker_id"], -1)
        self.assertEqual(turns[1]["text"], "Utterance 2")

    def test_repeated_distinct_voice_promotes_real_speaker(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [0, 1], 3, 5),
            row(3, 1, [.07, .997], 7, 9),
        ]
        stats = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(stats["confirmed"], 2)
        self.assertEqual([s["cluster"] for s in turns], [0, 1, 1])
        self.assertEqual(stats["deferred"], 0)

    def test_auditable_long_word_aligned_turn_can_confirm_new_voice(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [0, 1], 3, 7,
                timestamp_quality="word",
                words=[{"text": "a"}, {"text": "b"}, {"text": "c"}]),
        ]
        stats = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(stats["confirmed"], 2)

    def test_ambiguous_voice_not_assigned_to_another_person(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [.6, .8], 3, 5),
            row(3, 1, [.65, .76], 8, 10),
        ]
        stats = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(stats["deferred"], 2)
        self.assertEqual([s["cluster"] for s in turns], [0, -3, -3])

    def test_approved_manual_override_or_profile_is_not_changed(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [0, 1], 3, 5,
                speaker_override=True, speaker_id=9,
                speaker_name="Manually confirmed"),
            row(3, -1, [.99, .01], 7, 9,
                matched_profile_id="registered-1", speaker_id=10,
                speaker_name="Registered participant"),
        ]
        gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(turns[1]["speaker_id"], 9)
        self.assertEqual(turns[1]["speaker_name"], "Manually confirmed")
        self.assertEqual(turns[2]["speaker_id"], 10)

    def test_followup_reclustering_does_not_restore_old_phantom_label(self):
        turns = [
            row(1, 0, [1, 0], 0, 2, speaker_id=1, speaker_name="Speaker 1"),
            row(2, 1, [.99, .08], 3, 5,
                speaker_id=2, speaker_name="Speaker 2"),
        ]
        gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(turns[1]["speaker_id"], 1)
        self.assertEqual(turns[1]["speaker_name"], "Speaker 1")


if __name__ == "__main__":
    unittest.main()

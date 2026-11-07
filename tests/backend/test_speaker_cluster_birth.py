"""Prevent a single reader from becoming Speaker 1-6 in the default ASR path.

Synthetic vectors establish policy correctness, not real-world DER.
"""
from __future__ import annotations

import unittest
import numpy as np

from sessions.identity.speaker_cluster_birth import (gate_new_cluster_births, _bounded_references, MAX_ANCHOR_VECTORS)


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

    def test_continuous_reader_with_realistic_voice_variation_keeps_speaker_one(self):
        # One person speaks in changing volume and prosody with natural
        # 0-7 second pauses. Middle-confidence embeddings must not become
        # a dozen 'Unidentified Speaker' captions.
        turns = [
            row(1, 0, [1.0, 0.0], 3, 6),
            row(2, 1, [.72, .694], 13, 18),
            row(3, 2, [.70, .714], 18.3, 19.2),
            row(4, 3, [.69, .724], 20, 21.3),
            row(5, 4, [.73, .683], 23, 25),
            row(6, 5, [.71, .705], 27, 30),
        ]
        original = [
            (t["text"], t["start_sample"], t["end_sample"]) for t in turns
        ]
        results = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(results, {"reused": 5, "deferred": 0, "confirmed": 1})
        self.assertEqual([t["cluster"] for t in turns], [0] * 6)
        self.assertTrue(all(t.get("speaker_uncertain") for t in turns[1:]))
        self.assertTrue(all(
            t["speaker_evidence_status"] == "speaker_continuity_provisional"
            for t in turns[1:]
        ))
        self.assertEqual(
            [(t["text"], t["start_sample"], t["end_sample"]) for t in turns],
            original,
        )

    def test_continuity_does_not_merge_voice_after_long_absence(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [.70, .714], 35, 37),
        ]
        result = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(result, {"reused": 0, "deferred": 1, "confirmed": 1})
        self.assertEqual(turns[1]["cluster"], -3)

    def test_continuity_does_not_absorb_clear_second_person(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [0, 1], 3, 5),
            row(3, 1, [.04, .999], 7, 10),
            row(4, 2, [.66, .75], 10.5, 12),
        ]
        result = gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(result["confirmed"], 2)
        self.assertEqual([t["cluster"] for t in turns[:3]], [0, 1, 1])
        # Borderline vector has no clear advantage against the two real
        # voices. It must not hijack either established identity.
        self.assertEqual(turns[3]["cluster"], -3)

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

    def test_birth_gate_preserves_original_embedding_reason_for_diagnostics(self):
        turns = [
            row(1, 0, [1, 0], 0, 2),
            row(2, 1, [.6, .8], 3, 5),
        ]
        gate_new_cluster_births(turns, sample_rate=16000)
        self.assertEqual(
            turns[1]["speaker_upstream_evidence_status"], "turn_voiceprint"
        )
        self.assertEqual(
            turns[1]["speaker_birth_decision"], "speaker_birth_unconfirmed"
        )
        self.assertEqual(turns[1]["speaker_id"], -1)

    def test_long_meeting_voice_reference_pool_is_bounded(self):
        long_history = [
            np.asarray([1.0, .001 * i], dtype=np.float32)
            for i in range(1000)
        ]
        selected = _bounded_references(long_history)
        self.assertEqual(len(selected), MAX_ANCHOR_VECTORS)
        self.assertTrue(np.array_equal(selected[0], long_history[0]))
        self.assertTrue(np.array_equal(selected[-1], long_history[-1]))

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

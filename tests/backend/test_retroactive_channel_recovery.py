"""Tests for conservative backfill of earlier, unidentified Nemotron turns."""
from __future__ import annotations

import unittest

from sessions.retroactive_channel_recovery import backfill_confirmed_channels


class ChannelBackfillTests(unittest.TestCase):
    def setUp(self):
        self.projected = [{
            "id": 28, "speaker_id": -1,
            "speaker_diarization_source": "nemotron3_uncertain",
            "speaker_acoustic_channel": 3,
            "speaker_activity_status": "official_turn_primary",
            "speaker_activity_score": 0.98,
            "speaker_activity_margin": 0.98,
        }]
        self.original = [{
            "emb": [0.01, 1],
            "text": "Original transcript",
            "start_sample": 10, "end_sample": 12,
        }]

    def reconcile(self, matcher=lambda voice: 2):
        return backfill_confirmed_channels(
            self.projected, self.original, {3: 2}, matcher,
        )

    def test_restores_only_confirmed_channel_with_matching_voice(self):
        self.assertEqual(self.reconcile(), {28: 2})
        self.assertEqual(
            self.projected[0]["speaker_diarization_source"],
            "nemotron3_backfilled",
        )
        self.assertTrue(self.projected[0]["speaker_uncertain"])
        self.assertEqual(self.original[0]["text"], "Original transcript")
        self.assertEqual(self.original[0]["start_sample"], 10)

    def test_no_embedding_cannot_recover(self):
        self.original[0]["emb"] = None
        self.assertEqual(self.reconcile(), {})

    def test_strong_missing_embedding_on_corroborated_channel_is_provisional(self):
        self.original[0]["emb"] = None
        self.projected[0]["speaker_channel_gate_reason"] = "missing_voice_embedding"
        self.assertEqual(self.reconcile(), {28: 2})
        self.assertTrue(self.projected[0]["speaker_uncertain"])

    def test_conflicting_timeline_embedding_blocks_acoustic_backfill(self):
        self.original[0]["emb"] = None
        self.projected[0]["speaker_channel_gate_reason"] = "missing_voice_embedding"
        self.assertEqual(backfill_confirmed_channels(
            self.projected, self.original, {3: 2}, lambda voice: 1,
            effective_embeddings={28: [1, 0]},
        ), {})

    def test_wrong_voice_identity_cannot_recover(self):
        self.assertEqual(self.reconcile(lambda voice: 1), {})

    def test_ambiguous_activity_cannot_recover(self):
        self.projected[0]["speaker_activity_margin"] = 0.10
        self.assertEqual(self.reconcile(), {})

    def test_weak_activity_cannot_recover(self):
        self.projected[0]["speaker_activity_score"] = 0.45
        self.assertEqual(self.reconcile(), {})

    def test_no_stable_channel_mapping_cannot_recover(self):
        self.assertEqual(backfill_confirmed_channels(
            self.projected, self.original, {}, lambda voice: 2,
        ), {})

    def test_registered_or_manual_identity_is_unchanged(self):
        self.original[0]["speaker_override"] = True
        self.assertEqual(self.reconcile(), {})
        self.original[0]["speaker_override"] = False
        self.original[0]["matched_profile_id"] = "registered"
        self.assertEqual(self.reconcile(), {})

    def test_different_list_lengths_fail_closed(self):
        with self.assertRaises(ValueError):
            backfill_confirmed_channels(
                self.projected, [], {}, lambda voice: 2,
            )

    def test_cannot_create_speaker_from_channel_without_voice_match(self):
        self.assertEqual(self.reconcile(lambda voice: None), {})


if __name__ == "__main__":
    unittest.main()

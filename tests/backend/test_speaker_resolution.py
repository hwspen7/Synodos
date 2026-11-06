"""A missing Nemotron activity decision must not erase supported CAM++."""
from __future__ import annotations

import unittest
import numpy as np

from sessions.speaker_resolution import preserve_supported_camplus
from storage.meetings import MeetingStore


def record(**changes):
    base = {
        "speaker_id": 1, "speaker_name": "Speaker 1",
        "cluster": 0, "emb": np.asarray([1.0, 0.0], dtype=np.float32),
        "speaker_uncertain": False,
        "speaker_evidence_status": "turn_voiceprint",
    }
    base.update(changes)
    return base


class SpeakerResolutionTests(unittest.TestCase):
    def test_camplus_valid_voice_survives_missing_nemotron_activity(self):
        self.assertTrue(preserve_supported_camplus(record()))
        self.assertTrue(preserve_supported_camplus(record(
            speaker_evidence_status="acoustic_backfill"
        )))
        self.assertTrue(preserve_supported_camplus(record(
            speaker_evidence_status="multi_turn_same_voice_supported"
        )))

    def test_stated_continuity_remains_provisional_but_not_erased(self):
        self.assertTrue(preserve_supported_camplus(record(
            speaker_uncertain=True,
            speaker_evidence_status="speaker_continuity_provisional",
        )))

    def test_saved_evidence_diagnostics_never_expose_voiceprint_vectors(self):
        raw = record(
            id=15,
            start_sample=16000,
            end_sample=32000,
            speaker_upstream_evidence_status="multi_turn_acoustic_evidence_only",
            speaker_birth_decision="speaker_birth_unconfirmed",
        )
        safe = MeetingStore._segment_summary(raw, 16000)
        self.assertTrue(safe["speaker_embedding_available"])
        self.assertEqual(
            safe["speaker_upstream_evidence_status"],
            "multi_turn_acoustic_evidence_only",
        )
        self.assertEqual(safe["speaker_birth_decision"], "speaker_birth_unconfirmed")
        self.assertNotIn("emb", safe)

    def test_no_embedding_or_unsupported_cluster_stays_unknown(self):
        for change in (
            {"speaker_id": -1},
            {"cluster": -3},
            {"emb": None},
            {"emb": np.asarray([0, 0], dtype=np.float32)},
            {"speaker_evidence_status": "multi_turn_acoustic_evidence_only"},
            {"speaker_evidence_status": "short_probe", "speaker_uncertain": True},
            {"speaker_evidence_status": "speaker_birth_unconfirmed"},
            {"speaker_uncertain": True, "speaker_evidence_status": "turn_voiceprint"},
            {"speaker_override": True},
            {"matched_profile_id": "registered"},
        ):
            with self.subTest(change=str(change)):
                self.assertFalse(preserve_supported_camplus(record(**change)))


if __name__ == "__main__":
    unittest.main()

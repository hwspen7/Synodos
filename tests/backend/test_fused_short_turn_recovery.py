"""Regression tests for post-Nemotron, bidirectional short-turn recovery."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from sessions.channels.fused_short_turn_recovery import recover_fused_short_turns


SR = 16000


def segment(number, start, end, embedding, label, *,
            source="nemotron3", status="acoustic_primary", reason="turn_voiceprint"):
    original = {
        "id": number,
        "start_sample": round(start * SR),
        "end_sample": round(end * SR),
        "emb": None if embedding is None else np.asarray(embedding, dtype=np.float32),
        "speaker_evidence_status": reason,
        "text": f"Original transcript {number}",
    }
    projected = {
        "id": number,
        "speaker_id": label,
        "speaker_name": f"Speaker {label}" if label > 0 else "Unidentified Speaker",
        "speaker_diarization_source": source,
        "speaker_activity_status": status,
        "start_ms": round(start * 1000),
        "end_ms": round(end * 1000),
        "text": original["text"],
    }
    return original, projected


class FusedShortRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Unknowns occur before the second voice's later reference samples.
        specs = [
            segment(1, 20.0, 22.0, [1, 0], 1),
            segment(2, 30.0, 30.26, [0.999, 0.02], -1,
                    source="nemotron3_uncertain", status="acoustically_ambiguous",
                    reason="speaker_birth_unconfirmed"),
            segment(3, 35.0, 37.0, [.999, .02], 1),
            segment(4, 56.0, 56.28, [0.01, 1], -1,
                    source="nemotron3_uncertain", status="acoustically_ambiguous",
                    reason="speaker_birth_unconfirmed"),
            segment(5, 69.0, 71.0, [0, 1], 2),
            segment(6, 80.0, 83.0, [0.02, 0.999], 2),
        ]
        self.raw = [x[0] for x in specs]
        self.published = [x[1] for x in specs]

    def test_two_short_utterances_recovered_in_both_directions(self):
        originals = copy.deepcopy(self.raw)
        changes = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(changes, {2: 1, 4: 2})
        self.assertEqual(self.published[1]["speaker_status"],
                         "short_voice_recovered_provisional")
        self.assertTrue(self.published[1]["speaker_uncertain"])
        self.assertEqual(self.published[3]["speaker_diarization_source"],
                         "camplus_bidirectional_recovery")
        self.assertEqual([x["text"] for x in self.raw],
                         [x["text"] for x in originals])
        self.assertEqual([x["start_sample"] for x in self.raw],
                         [x["start_sample"] for x in originals])
        self.assertEqual([x["end_sample"] for x in self.raw],
                         [x["end_sample"] for x in originals])

    def test_missing_embedding_stays_unknown(self):
        self.raw[1]["emb"] = None
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {4: 2})
        self.assertEqual(self.published[1]["speaker_id"], -1)

    def test_ambiguous_or_noisy_short_voice_stays_unknown(self):
        self.raw[1]["emb"] = np.asarray([.707, .707], dtype=np.float32)
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {4: 2})

    def test_single_reference_does_not_create_confidence(self):
        self.published[5]["speaker_diarization_source"] = "camplus_provisional"
        results = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(results, {2: 1})

    def test_raw_camplus_label_is_not_trusted_as_anchor(self):
        self.published[4]["speaker_diarization_source"] = "camplus_provisional"
        self.published[5]["speaker_diarization_source"] = "camplus_provisional"
        results = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(results, {2: 1})

    def test_repeat_call_cannot_self_reinforce(self):
        changes = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(changes, {2: 1, 4: 2})
        second = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(second, {})

    def test_old_unconfirmed_non_short_turn_is_not_forced(self):
        self.raw[1]["end_sample"] = round(32.1 * SR)
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {4: 2})

    def test_manual_override_remains_unchanged(self):
        self.raw[1]["speaker_override"] = True
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {4: 2})

    def test_official_turn_primary_is_eligible_as_reference(self):
        self.published[0]["speaker_activity_status"] = "official_turn_primary"
        self.published[2]["speaker_activity_status"] = "official_turn_primary"
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {2: 1, 4: 2})

    def test_cross_voice_overlap_is_not_used_as_reference(self):
        self.raw[4]["start_sample"] = round(55.9 * SR)
        self.raw[4]["end_sample"] = round(58.0 * SR)
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {2: 1})

    def test_nemotron_ambiguous_turn_not_used_as_reference(self):
        self.published[5]["speaker_activity_status"] = "acoustically_ambiguous"
        self.assertEqual(recover_fused_short_turns(
            self.published, self.raw, sample_rate=SR), {2: 1})

    def test_bad_vector_and_mismatched_dimension_stay_unknown(self):
        self.raw[1]["emb"] = np.asarray([float("nan"), 1.0])
        self.raw[2]["emb"] = np.asarray([1.0, 0, 0], dtype=np.float32)
        results = recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(results, {4: 2})

    def test_uncertain_unknown_keeps_original_activity_evidence(self):
        recover_fused_short_turns(self.published, self.raw, sample_rate=SR)
        self.assertEqual(self.published[1]["speaker_activity_status"],
                         "acoustically_ambiguous")
        self.assertEqual(self.raw[1]["speaker_evidence_status"],
                         "speaker_birth_unconfirmed")


if __name__ == "__main__":
    unittest.main()

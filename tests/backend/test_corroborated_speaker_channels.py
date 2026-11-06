"""Regression tests for phantom speakers in one uninterrupted monologue."""
from __future__ import annotations

import unittest
import numpy as np

from sessions.corroborated_speaker_channels import CorroboratedSpeakerChannels


def vec(x: float, y: float) -> np.ndarray:
    result = np.asarray([x, y], dtype=np.float32)
    return result / np.linalg.norm(result)


class CorroboratedSpeakerTests(unittest.TestCase):
    def test_single_voice_five_raw_channels_remains_one_speaker(self):
        guard = CorroboratedSpeakerChannels()
        # Nemotron invents different channels after each sentence, but the
        # independent voice embeddings agree that this is one human voice.
        clips = [
            (1, 101, 1500, vec(1, 0)),
            (2, 102, 1400, vec(.98, .20)),
            (3, 103, 1800, vec(.96, .28)),
            (4, 104, 1600, vec(.99, -.12)),
            (5, 105, 2100, vec(.93, .36)),
        ]
        self.assertEqual(
            [guard.observe(channel, sid, ms, emb)
             for channel, sid, ms, emb in clips],
            [1, 1, 1, 1, 1],
        )
        self.assertEqual(set(guard.labels.values()), {1})

    def test_no_embedding_does_not_create_a_second_speaker(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(5, 1, 1400, vec(1, 0)), 1)
        self.assertIsNone(guard.observe(2, 2, 2500, None))
        self.assertEqual(guard.labels, {5: 1})

    def test_medium_distance_voice_requires_sustained_consensus(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(1, 1, 1700, vec(1, 0)), 1)
        results = [
            guard.observe(2, seg_id, 2000, vec(.62, .7846))
            for seg_id in range(2, 7)
        ]
        self.assertEqual(results, [None, None, None, None, 2])
        self.assertEqual(guard.gate_reason(2), "medium_voice_confirmed")
        self.assertEqual(guard.labels, {1: 1, 2: 2})

    def test_medium_distance_voice_still_unknown_without_five_turns(self):
        guard = CorroboratedSpeakerChannels()
        guard.observe(1, 1, 1700, vec(1, 0))
        for seg_id in range(2, 6):
            self.assertIsNone(guard.observe(2, seg_id, 2000, vec(.62, .7846)))
        self.assertEqual(guard.labels, {1: 1})
        self.assertEqual(guard.gate_reason(2), "candidate_needs_more_turns")

    def test_too_similar_never_born_as_new_even_after_many_segments(self):
        guard = CorroboratedSpeakerChannels()
        guard.observe(1, 1, 1700, vec(1, 0))
        for seg_id in range(2, 17):
            self.assertIsNone(guard.observe(2, seg_id, 2500, vec(.74, .672)))
        self.assertEqual(guard.labels, {1: 1})
        self.assertEqual(
            guard.gate_reason(2), "candidate_similar_to_existing_voice"
        )

    def test_mixed_candidate_cannot_bypass_medium_consensus(self):
        guard = CorroboratedSpeakerChannels()
        guard.observe(1, 1, 1700, vec(1, 0))
        vectors = [
            vec(.65, .76), vec(0, 1), vec(.65, .76), vec(0, 1),
            vec(.65, .76), vec(0, 1),
        ]
        results = [
            guard.observe(2, seg_id, 2000, voice)
            for seg_id, voice in enumerate(vectors, start=2)
        ]
        self.assertTrue(all(result is None for result in results))
        self.assertEqual(guard.labels, {1: 1})

    def test_ambiguous_similarity_does_not_create_new_identity(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(1, 1, 1600, vec(1, 0)), 1)
        self.assertIsNone(guard.observe(3, 2, 2100, vec(.65, .76)))
        self.assertIsNone(guard.observe(3, 3, 2300, vec(.64, .77)))
        self.assertEqual(guard.labels, {1: 1})

    def test_distinct_voice_requires_repeated_turns(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(1, 1, 1800, vec(1, 0)), 1)
        self.assertIsNone(guard.observe(2, 2, 1700, vec(0, 1)))
        self.assertIsNone(guard.observe(2, 2, 1700, vec(0, 1)))
        self.assertEqual(guard.labels, {1: 1})
        self.assertEqual(guard.observe(2, 3, 1600, vec(.05, .998)), 2)
        # On a subsequent rebuild the first candidate turn is now published.
        self.assertEqual(guard.observe(2, 2, 1700, vec(0, 1)), 2)

    def test_ambiguous_different_voice_samples_do_not_prove_new_person(self):
        guard = CorroboratedSpeakerChannels()
        guard.observe(1, 1, 1700, vec(1, 0))
        self.assertIsNone(guard.observe(2, 2, 1700, vec(0, 1)))
        # A second sample for the supposed new person conflicts with the
        # first sample and is therefore not independent corroboration.
        self.assertIsNone(guard.observe(2, 3, 1700, vec(0, -1)))
        self.assertEqual(guard.labels, {1: 1})

    def test_same_voice_reuse_does_not_consume_a_new_label(self):
        guard = CorroboratedSpeakerChannels(offset=4)
        self.assertEqual(guard.observe(5, 1, 1300, vec(1, 0)), 5)
        self.assertEqual(guard.observe(7, 2, 1700, vec(.99, .04)), 5)
        self.assertIsNone(guard.observe(2, 3, 1700, vec(0, 1)))
        self.assertEqual(guard.observe(2, 4, 1600, vec(.04, .99)), 6)
        self.assertEqual(guard.labels, {5: 5, 7: 5, 2: 6})

    def test_reuse_known_voice_without_relying_on_nemotron_channel(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(1, 1, 1700, vec(1, 0)), 1)
        self.assertEqual(guard.match_known(vec(.99, .10)), 1)
        self.assertIsNone(guard.match_known(vec(0, 1)))
        self.assertIsNone(guard.match_known(None))
        self.assertEqual(guard.labels, {1: 1})

    def test_same_channel_reused_for_other_voice_is_not_misattributed(self):
        guard = CorroboratedSpeakerChannels()
        self.assertEqual(guard.observe(1, 1, 1800, vec(1, 0)), 1)
        self.assertIsNone(guard.observe(1, 2, 1800, vec(0, 1)))
        self.assertEqual(guard.labels, {1: 1})

    def test_short_phantom_and_wrong_type_fail_closed(self):
        guard = CorroboratedSpeakerChannels()
        self.assertIsNone(guard.observe(1, 1, 200, vec(1, 0)))
        self.assertEqual(guard.observe(1, 1, 1800, vec(1, 0)), 1)
        for bad_channel in (-1, 0, 9, True):
            self.assertIsNone(guard.observe(bad_channel, 10, 2400, vec(1, 0)))
        self.assertEqual(guard.labels, {1: 1})


if __name__ == "__main__":
    unittest.main()

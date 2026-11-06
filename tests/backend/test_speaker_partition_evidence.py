"""Acoustic word-group evidence must not create phantom human identities."""
from __future__ import annotations

import unittest
import numpy as np

from sessions.speaker_partition_evidence import consistent_partition_embedding


def part(start, end, voice, **overrides):
    return {
        "start": start, "end": end, "text": "spoken text",
        "emb": np.asarray(voice, dtype=np.float32) if voice is not None else None,
        "ambiguous": False, **overrides,
    }


class PartitionEvidenceTests(unittest.TestCase):
    def test_single_voice_across_multiple_acoustic_cuts_is_recovered(self):
        result = consistent_partition_embedding([
            part(0, 1500, [1.0, 0.0]),
            part(1500, 3000, [.99, .05]),
            part(3000, 4900, [.97, -.11]),
        ])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_distinct_speakers_never_share_one_embedding(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1.0, 0]),
            part(1500, 3000, [0, 1.0]),
        ]))

    def test_unknown_or_short_piece_fails_closed(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 400, [1, 0]),
            part(400, 2200, [1, 0]),
        ]))
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1, 0]),
            part(1500, 3000, None),
        ]))
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1, 0]),
            part(1500, 3000, [1, 0], ambiguous=True),
        ]))
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1, 0]),
            part(1500, 3000, [1, 0], short_probe=True),
        ]))

    def test_eight_second_single_voice_recovers_despite_one_short_missing_piece(self):
        chunks = [
            part(0, 2600, [1, 0]),
            part(2600, 2950, None, short_probe=True),
            part(2950, 5600, [.99, .10]),
            part(5600, 8000, [.98, -.12]),
        ]
        embedding = consistent_partition_embedding(chunks)
        self.assertIsNotNone(embedding)
        self.assertGreater(float(embedding[0]), .98)

    def test_one_short_real_other_voice_blocks_single_person_claim(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 2500, [1, 0]),
            part(2500, 3100, [0, 1], short_probe=True),
            part(3100, 6100, [.99, .05]),
        ]))

    def test_long_unknown_voice_gap_is_not_silently_absorbed(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 2900, [1, 0]),
            part(2900, 4400, None),
            part(4400, 7500, [.99, .05]),
        ]))

    def test_one_valid_piece_with_missing_subturn_remains_unknown(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 3100, [1, 0]),
            part(3100, 3500, None, short_probe=True),
        ]))

    def test_two_distinct_long_voices_in_same_asr_turn_rejected(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 3200, [1, 0]),
            part(3200, 3700, None, short_probe=True),
            part(3700, 7600, [0, 1]),
        ]))

    def test_embedding_dimension_mismatch_does_not_raise(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1, 0]),
            part(1500, 3000, [1, 0, 0]),
        ]))


if __name__ == "__main__":
    unittest.main()

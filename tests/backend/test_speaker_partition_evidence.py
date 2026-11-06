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

    def test_embedding_dimension_mismatch_does_not_raise(self):
        self.assertIsNone(consistent_partition_embedding([
            part(0, 1500, [1, 0]),
            part(1500, 3000, [1, 0, 0]),
        ]))


if __name__ == "__main__":
    unittest.main()

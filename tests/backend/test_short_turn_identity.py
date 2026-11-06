"""Regression: don't leave one continuous voice unidentified after Speaker 1."""
from __future__ import annotations

import unittest
import numpy as np

from sessions.channels.short_turn_identity import recover_short_turn_identity


def clip(ident, label, start, end, vector, *, short=False, **fields):
    return {
        "id": ident,
        "start_sample": round(start * 16000),
        "end_sample": round(end * 16000),
        "speaker_id": label,
        "speaker_name": f"Speaker {label}" if label > 0 else "Unidentified Speaker",
        "speaker_uncertain": short,
        "speaker_evidence_status": "short_probe" if short else "turn_voiceprint",
        "emb": np.asarray(vector, dtype=np.float32) if vector is not None else None,
        "text": f"Original sentence {ident}",
        **fields,
    }


class ShortTurnIdentityTests(unittest.TestCase):
    def test_one_person_continues_across_short_utterance(self):
        rows = [
            clip(1, 1, 3, 6, [1, 0]),
            clip(2, -1, 13, 13.9, [.98, .12], short=True),
            clip(3, -1, 14, 14.6, [.97, .16], short=True),
        ]
        original = [(s["id"], s["text"], s["start_sample"], s["end_sample"])
                    for s in rows]
        count = recover_short_turn_identity(rows, sample_rate=16000)
        self.assertEqual(count, 2)
        self.assertEqual([s["speaker_id"] for s in rows], [1, 1, 1])
        self.assertTrue(rows[1]["speaker_uncertain"])
        self.assertEqual(rows[1]["speaker_evidence_status"], "short_probe_known_voice")
        self.assertEqual(
            [(s["id"], s["text"], s["start_sample"], s["end_sample"])
             for s in rows], original,
        )
        # Rebuilding the same result must not continually relabel again.
        self.assertEqual(recover_short_turn_identity(rows, sample_rate=16000), 2)

    def test_cannot_invent_speaker_or_transfer_different_voice(self):
        rows = [
            clip(1, 1, 0, 3, [1, 0]),
            clip(2, -1, 4, 4.7, [0, 1], short=True),
        ]
        self.assertEqual(recover_short_turn_identity(rows, sample_rate=16000), 0)
        self.assertEqual(rows[1]["speaker_id"], -1)

    def test_no_embedding_or_long_gap_remains_unknown(self):
        for other in [
            clip(2, -1, 3, 3.5, None, short=True),
            clip(2, -1, 20, 20.5, [1, 0], short=True),
        ]:
            rows = [clip(1, 1, 0, 3, [1, 0]), other]
            self.assertEqual(recover_short_turn_identity(rows, sample_rate=16000), 0)
            self.assertEqual(other["speaker_id"], -1)

    def test_two_known_speakers_need_clear_margin(self):
        rows = [
            clip(1, 1, 0, 2, [1, 0]),
            clip(2, 2, 3, 5, [.92, .392]),
            clip(3, -1, 5.5, 6, [.98, .2], short=True),
        ]
        self.assertEqual(recover_short_turn_identity(rows, sample_rate=16000), 0)
        self.assertEqual(rows[-1]["speaker_id"], -1)

    def test_proven_identity_is_never_modified(self):
        rows = [
            clip(1, 1, 0, 2, [1, 0]),
            clip(2, 7, 3, 4, [.99, .1], short=True,
                 speaker_override=True, speaker_override_id=7),
        ]
        self.assertEqual(recover_short_turn_identity(rows, sample_rate=16000), 0)
        self.assertEqual(rows[-1]["speaker_id"], 7)


if __name__ == "__main__":
    unittest.main()

"""Default CAM++ pathway must not publish five people from one monologue."""
from __future__ import annotations

import unittest
import numpy as np

from sessions.rapid_speaker_fragmentation import repair_rapid_anonymous_births


def segment(sid, speaker_id, t0, t1, vector, **extra):
    return {
        "id": sid,
        "start_sample": t0 * 16000,
        "end_sample": t1 * 16000,
        "speaker_id": speaker_id,
        "speaker_name": f"Speaker {speaker_id}",
        "cluster": speaker_id - 1,
        "emb": np.asarray(vector, dtype=np.float32) if vector is not None else None,
        "text": f"Utterance {sid}",
        **extra,
    }


class RapidSpeakerFragmentationTests(unittest.TestCase):
    def test_five_camplus_clusters_same_person_reuse_one_display_id(self):
        rows = [
            segment(1, 1, 0, 3, [1, 0]),
            segment(2, 2, 3, 5, [.98, .13]),
            segment(3, 3, 5, 7, [.97, .15]),
            segment(4, 4, 8, 10, [.96, -.15]),
            segment(5, 5, 10, 12, [.95, .18]),
        ]
        before = [(s["id"], s["text"], s["start_sample"], s["end_sample"])
                  for s in rows]
        audit = repair_rapid_anonymous_births(rows, sample_rate=16000)
        self.assertEqual(audit, {"merged": 4, "deferred": 0})
        self.assertEqual([s["speaker_id"] for s in rows], [1] * 5)
        self.assertEqual(
            [(s["id"], s["text"], s["start_sample"], s["end_sample"])
             for s in rows],
            before,
        )

    def test_unverified_short_clusters_stay_unknown_without_voiceproof(self):
        rows = [
            segment(1, 1, 0, 3, [1, 0]),
            segment(2, 2, 3, 5, None),
            segment(3, 3, 5, 7, [.6, .8]),
        ]
        result = repair_rapid_anonymous_births(rows, sample_rate=16000)
        self.assertEqual(result, {"merged": 0, "deferred": 2})
        # No evidence is not evidence of an incorrect identity. Preserve
        # existing provisional labels until actual voice evidence arrives.
        self.assertEqual([s["speaker_id"] for s in rows], [1, 2, 3])
        self.assertTrue(all(s["speaker_uncertain"] for s in rows[1:]))
        self.assertTrue(all(
            s["speaker_evidence_status"] == "rapid_cluster_unconfirmed"
            for s in rows[1:]
        ))

    def test_two_real_speakers_are_not_filtered(self):
        rows = [
            segment(1, 1, 0, 3, [1, 0]),
            segment(2, 2, 3, 5, [0, 1]),
            segment(3, 1, 7, 10, [1, .03]),
            segment(4, 2, 11, 14, [.03, 1]),
        ]
        self.assertEqual(
            repair_rapid_anonymous_births(rows, sample_rate=16000),
            {"merged": 0, "deferred": 0},
        )
        self.assertEqual([s["speaker_id"] for s in rows], [1, 2, 1, 2])

    def test_repeated_distinct_third_voice_is_preserved(self):
        rows = [
            segment(1, 1, 0, 2, [1, 0, 0]),
            segment(2, 2, 3, 5, [0, 1, 0]),
            segment(3, 3, 6, 8, [0, 0, 1]),
            segment(4, 3, 10, 12, [.01, 0, 1]),
        ]
        # The first and second one-off labels may remain uncertain, but the
        # third voice has two independent observations and is not merged.
        repair_rapid_anonymous_births(rows, sample_rate=16000)
        self.assertEqual([s["speaker_id"] for s in rows[-2:]], [3, 3])

    def test_manual_and_registered_identities_never_change(self):
        rows = [
            segment(1, 1, 0, 3, [1, 0]),
            segment(2, 2, 3, 5, [1, 0], speaker_override=True),
            segment(3, 3, 5, 7, [1, 0], matched_profile_id="saved-voice"),
            segment(4, 4, 7, 9, [1, 0]),
        ]
        repair_rapid_anonymous_births(rows, sample_rate=16000)
        self.assertEqual(rows[1]["speaker_id"], 2)
        self.assertEqual(rows[2]["speaker_id"], 3)
        self.assertEqual(rows[3]["speaker_id"], 4)

    def test_source_only_singleton_has_no_global_effect(self):
        rows = [segment(1, 1, 0, 3, [1, 0])]
        self.assertEqual(
            repair_rapid_anonymous_births(rows, sample_rate=16000),
            {"merged": 0, "deferred": 0},
        )


if __name__ == "__main__":
    unittest.main()

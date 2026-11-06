"""Phase 07C turn diagnostics."""
import unittest
from runtime_quality.speakers import speaker_stats, find_distinct_participant


class SpeakerDiagnosticsTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(speaker_stats([])["speaker_switches"], 0)

    def test_consecutive_same_speaker(self):
        rows = [{"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
                {"start_ms": 1500, "end_ms": 2000, "speaker_id": 1}]
        self.assertEqual(speaker_stats(rows)["speaker_switches"], 0)

    def test_real_switches(self):
        rows = [{"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "speaker_id": sid}
                for i, sid in enumerate([1, 2, 2, 1])]
        self.assertEqual(speaker_stats(rows)["speaker_switches"], 2)

    def test_unknown_does_not_invent_switch(self):
        rows = [{"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "speaker_id": sid}
                for i, sid in enumerate([1, -1, 1])]
        self.assertEqual(speaker_stats(rows)["speaker_switches"], 0)
        self.assertEqual(speaker_stats(rows)["unidentified_segments"], 1)

    def test_overlap_only_measures_timing(self):
        rows = [{"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
                {"start_ms": 800, "end_ms": 2000, "speaker_id": 2}]
        self.assertEqual(speaker_stats(rows)["timestamp_overlap_ms"], 200)


class SpeakerMarginTests(unittest.TestCase):
    def test_clear_winner(self):
        import numpy as np
        participants = [{"id": "a", "vectors": [{"emb": np.array([1., 0.])}]},
                        {"id": "b", "vectors": [{"emb": np.array([0., 1.])}]}]
        self.assertEqual(find_distinct_participant(participants, np.array([1., 0.]), .03)[0], "a")

    def test_tied_speakers_unknown(self):
        import numpy as np
        participants = [{"id": "a", "vectors": [{"emb": np.array([1., 0.])}]},
                        {"id": "b", "vectors": [{"emb": np.array([.999, 0.])}]}]
        self.assertIsNone(find_distinct_participant(participants, np.array([1., 0.]), .03)[0])

    def test_same_person_multiple_vectors_not_a_tie(self):
        import numpy as np
        participants = [{"id": "a", "vectors": [{"emb": np.array([1., 0.])},
                                               {"emb": np.array([.99, 0.])}]},
                        {"id": "b", "vectors": [{"emb": np.array([0., 1.])}]}]
        self.assertEqual(find_distinct_participant(participants, np.array([1., 0.]), .1)[0], "a")

    def test_empty_participants(self):
        import numpy as np
        self.assertEqual(find_distinct_participant([], np.array([1., 0.]), .03), (None, -1.))

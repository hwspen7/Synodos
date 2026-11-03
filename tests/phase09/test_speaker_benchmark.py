"""Speaker DER gates must not reward unknown or randomly renamed speakers."""
import unittest

from tools.phase09_speaker_benchmark import evaluate


class SpeakerBenchmarkTests(unittest.TestCase):
    def test_permutation_invariant_identity(self):
        reference = [
            {"start_ms": 0, "end_ms": 2000, "speaker": "A"},
            {"start_ms": 2000, "end_ms": 4000, "speaker": "B"},
        ]
        hypothesis = [
            {"start_ms": 0, "end_ms": 2000, "speaker_id": 9},
            {"start_ms": 2000, "end_ms": 4000, "speaker_id": 3},
        ]
        self.assertEqual(evaluate(reference, hypothesis)["der"], 0.0)

    def test_unknown_is_not_assumed_to_be_a_person(self):
        reference = [{"start_ms": 0, "end_ms": 1000, "speaker": "A"}]
        hypothesis = [{"start_ms": 0, "end_ms": 1000, "speaker_id": -1}]
        self.assertEqual(evaluate(reference, hypothesis)["der"], 1.0)

    def test_false_positive_speaking_counts_as_error(self):
        reference = [{"start_ms": 0, "end_ms": 1000, "speaker": "A"}]
        hypothesis = [{"start_ms": 0, "end_ms": 2000, "speaker_id": 1}]
        self.assertEqual(evaluate(reference, hypothesis)["der"], 1.0)

    def test_overlap_requires_both_speakers(self):
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 0, "end_ms": 1000, "speaker": "B"},
        ]
        hypothesis = [{"start_ms": 0, "end_ms": 1000, "speaker_id": 1}]
        self.assertEqual(evaluate(reference, hypothesis)["der"], 0.5)

    def test_rejects_unlabeled_reference(self):
        with self.assertRaises(ValueError):
            evaluate([], [])


if __name__ == "__main__":
    unittest.main()

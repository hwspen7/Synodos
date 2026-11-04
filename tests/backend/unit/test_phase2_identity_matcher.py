"""Identity matching must abstain rather than invent enrolled speakers."""
import unittest

import numpy as np

from sessions.identity_matcher import match_enrolled_voice


class IdentityMatcherTests(unittest.TestCase):
    def test_distinct_enrollment_and_model_specificity(self):
        profiles = [
            {"id": "alice", "vectors": [{"emb": [1., 0.],
                "embedding_model": "camplus", "short": False}]},
            {"id": "bob", "vectors": [{"emb": [0., 1.],
                "embedding_model": "camplus", "short": False}]},
        ]
        found = match_enrolled_voice(profiles, np.array([1., 0.]),
                                     threshold=.6, margin=.05)
        self.assertEqual(found["profile_id"], "alice")
        self.assertEqual(found["status"], "matched")
        self.assertEqual(found["score_kind"], "raw_cosine_not_probability")
        self.assertEqual(
            match_enrolled_voice(profiles, [1., 0.], model_id="ecapa")["status"],
            "no_compatible_enrollment",
        )

    def test_ambiguity_fails_closed(self):
        profiles = [
            {"id": "alice", "vectors": [{"emb": [1., 0.]}]},
            {"id": "bob", "vectors": [{"emb": [.999, .001]}]},
        ]
        result = match_enrolled_voice(profiles, [1., 0.], margin=.06)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["profile_id"])

    def test_malformed_and_short_vectors_are_ignored(self):
        people = [
            {"id": "invalid", "vectors": [
                {"emb": [float("nan"), 1.]},
                {"emb": [1., 0.], "short": True},
            ]},
            {"id": "other_dim", "vectors": [{"emb": [1., 0., 0.]}]},
        ]
        result = match_enrolled_voice(people, [1., 0.])
        self.assertIsNone(result["profile_id"])
        self.assertEqual(result["status"], "no_compatible_enrollment")
        self.assertEqual(
            match_enrolled_voice(people, [0., 0.])["status"], "invalid_probe"
        )

    def test_low_similarity_not_an_identity(self):
        profile = [{"id": "alice", "vectors": [{"emb": [0., 1.]}]}]
        result = match_enrolled_voice(profile, [1., 0.], threshold=.7)
        self.assertEqual(result["status"], "below_threshold")
        self.assertIsNone(result["profile_id"])


if __name__ == "__main__":
    unittest.main()

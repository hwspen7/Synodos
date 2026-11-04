"""Trial builder uses only explicitly reviewed embeddings, never names."""
import unittest

import numpy as np

from tools.phase2_build_trials import build_trials, eligible_vectors
from tools.phase2_cross_session_acceptance import validate_trials


def profiles():
    rows = []
    for person, vector in (
        ("a", np.array([1.0, 0.0] + [0.0] * 30)),
        ("b", np.array([0.0, 1.0] + [0.0] * 30)),
    ):
        rows.append({
            "id": person,
            "name": "not in output",
            "vectors": [
                {
                    "id": f"{person}-{session}-{index}",
                    "emb": vector,
                    "meeting_id": session,
                    "embedding_model": "camplus",
                    "human_verified": True,
                    "verification_source": "manual_profile_correction",
                    "short": False,
                }
                for session in ("m1", "m2", "m3")
                for index in range(4)
            ]
        })
    return rows


class BuildPhase2TrialsTests(unittest.TestCase):
    def test_explicit_verified_human_vectors_generate_true_cross_session_pairs(self):
        output = build_trials(
            profiles(), model_id="camplus",
            meetings={"m1", "m2", "m3"},
        )
        self.assertGreaterEqual(sum(x["same_person"]
                                    for x in output["trials"]), 20)
        self.assertGreaterEqual(sum(not x["same_person"]
                                    for x in output["trials"]), 20)
        validate_trials(output)
        self.assertNotIn("not in output", str(output))
        self.assertNotIn("emb", output["trials"][0])

    def test_unverified_and_wrong_embedding_model_are_excluded(self):
        data = profiles()
        for person in data:
            for vector in person["vectors"]:
                vector["human_verified"] = False
        self.assertEqual(
            eligible_vectors(data, model_id="camplus",
                             meetings={"m1", "m2", "m3"}), [],
        )
        for person in data:
            for vector in person["vectors"]:
                vector["human_verified"] = True
        self.assertEqual(
            eligible_vectors(data, model_id="ecapa",
                             meetings={"m1", "m2", "m3"}), [],
        )

    def test_no_fabricated_pairs_on_too_little_recording_evidence(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_trials(profiles(), model_id="camplus",
                         meetings={"m1"})


if __name__ == "__main__":
    unittest.main()

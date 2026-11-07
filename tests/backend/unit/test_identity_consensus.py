"""Speaker consensus is never an automatic identity mutation."""
import unittest

from sessions.identity.identity_consensus import reconcile_identity_decisions


def matched(model, profile):
    return {
        "status": "matched",
        "embedding_model": model,
        "profile_id": profile,
        "score_kind": "raw_cosine_not_probability",
        "enrollment_human_verified": True,
    }


class ConsensusTests(unittest.TestCase):
    def test_matching_independent_networks_agree(self):
        record = reconcile_identity_decisions({
            "camplus": matched("camplus", "person-A"),
            "ecapa": matched("ecapa", "person-A"),
        })
        self.assertEqual(record["status"], "two_model_consensus")
        self.assertEqual(record["profile_id"], "person-A")
        self.assertFalse(record["automatic_identity_update"])
        self.assertFalse(record["raw_scores_combined"])

    def test_disagreement_is_unknown_not_best_cosine(self):
        record = reconcile_identity_decisions({
            "camplus": matched("camplus", "person-A"),
            "ecapa": matched("ecapa", "person-B"),
        })
        self.assertEqual(record["status"], "model_disagreement")
        self.assertIsNone(record["profile_id"])

    def test_two_unverified_cosine_matches_cannot_claim_verified_consensus(self):
        a = matched("camplus", "person-A")
        b = matched("ecapa", "person-A")
        a["enrollment_human_verified"] = False
        b["enrollment_human_verified"] = False
        outcome = reconcile_identity_decisions({"camplus": a, "ecapa": b})
        self.assertEqual(outcome["status"], "insufficient_evidence")
        self.assertIsNone(outcome["profile_id"])
        self.assertEqual(outcome["model_status"]["camplus"], "unverified_enrollment")
        self.assertFalse(outcome["automatic_identity_update"])

    def test_one_verified_one_unverified_is_not_two_model_consensus(self):
        a = matched("camplus", "person-A")
        b = matched("ecapa", "person-A")
        b["enrollment_human_verified"] = False
        outcome = reconcile_identity_decisions({"camplus": a, "ecapa": b})
        self.assertEqual(outcome["status"], "single_model_only")
        self.assertIsNone(outcome["profile_id"])

    def test_single_network_never_counts_as_two_model_consensus(self):
        record = reconcile_identity_decisions({
            "camplus": matched("camplus", "person-A"),
        })
        self.assertEqual(record["status"], "single_model_only")
        self.assertIsNone(record["profile_id"])

    def test_missing_or_bad_identity_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            reconcile_identity_decisions({
                "camplus": matched("ecapa", "person-A"),
            })
        with self.assertRaises(ValueError):
            reconcile_identity_decisions({
                "camplus": {
                    "status": "matched",
                    "embedding_model": "camplus",
                    "profile_id": None,
                }
            })

    def test_empty_evidence_remains_unknown(self):
        record = reconcile_identity_decisions({})
        self.assertEqual(record["status"], "insufficient_evidence")
        self.assertIsNone(record["profile_id"])


if __name__ == "__main__":
    unittest.main()

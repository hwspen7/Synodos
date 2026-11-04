"""Cross-meeting identity acceptance tests without embedding or personal data."""
import copy
import unittest

from tools.phase2_cross_session_acceptance import (
    SCHEMA, evaluate, rates, validate_trials,
)


def trials(prefix, *, model="camplus"):
    records = []
    for i in range(20):
        for same in (True, False):
            records.append({
                "id": f"{prefix}_{i}_{int(same)}",
                "session_a": f"{prefix}a",
                "session_b": f"{prefix}b" if i % 2 else f"{prefix}c",
                "person_group": f"{prefix}-person-group",
                "embedding_model": model,
                "human_labeled": True,
                "same_person": same,
                "cosine": 0.8 if same else 0.1,
            })
    return {"schema": SCHEMA, "trials": records}


class Phase2CrossSessionTests(unittest.TestCase):
    def test_calibrates_on_dev_without_holdout_label_leakage(self):
        result = evaluate(trials("dev"), trials("heldout"))
        self.assertTrue(result["holdout_goal_met"])
        self.assertEqual(result["dev_pairs"], 40)
        self.assertEqual(result["holdout_pairs"], 40)
        self.assertEqual(result["holdout"]["false_accepts"], 0)
        self.assertFalse(result["may_auto_update_production_threshold"])

    def test_scores_impostors_and_genuine_separately(self):
        data = validate_trials(trials("training"))
        metrics = rates(data, 0.9)
        self.assertEqual(metrics["false_accepts"], 0)
        self.assertEqual(metrics["false_rejects"], 20)

    def test_development_holdout_session_overlap_fails(self):
        dev, heldout = trials("same"), trials("other")
        heldout["trials"][0]["session_a"] = "samea"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate(dev, heldout)

    def test_development_holdout_person_overlap_fails(self):
        dev, heldout = trials("dev"), trials("heldout")
        heldout["trials"][0]["person_group"] = "dev-person-group"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate(dev, heldout)

    def test_rejects_unlabeled_same_session_and_model_mixing(self):
        bad = trials("cohort")
        bad["trials"][0]["human_labeled"] = False
        with self.assertRaisesRegex(ValueError, "human"):
            validate_trials(bad)
        bad = trials("cohort")
        bad["trials"][0]["session_b"] = "cohorta"
        with self.assertRaises(ValueError):
            validate_trials(bad)
        bad = trials("cohort")
        bad["trials"][0]["embedding_model"] = "ecapa"
        with self.assertRaisesRegex(ValueError, "models"):
            validate_trials(bad)


if __name__ == "__main__":
    unittest.main()

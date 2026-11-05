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
                "speaker_a": f"{prefix}-subject-1",
                "speaker_b": (
                    f"{prefix}-subject-1" if same else f"{prefix}-subject-2"
                ),
                "clip_a": f"{prefix}-clip-a-{i}-{int(same)}",
                "clip_b": f"{prefix}-clip-b-{i}-{int(same)}",
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
        heldout["trials"][0]["speaker_a"] = "dev-subject-1"
        heldout["trials"][0]["speaker_b"] = "dev-subject-1"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate(dev, heldout)

    def test_duplicate_clip_pairs_rejected(self):
        sample = trials("repeated")
        sample["trials"][1]["clip_a"] = sample["trials"][0]["clip_a"]
        sample["trials"][1]["clip_b"] = sample["trials"][0]["clip_b"]
        with self.assertRaisesRegex(ValueError, "Duplicate source clips"):
            validate_trials(sample)

    def test_reused_clip_cannot_change_its_meeting_or_human_label(self):
        sample = trials("provenance")
        sample["trials"][1]["clip_a"] = sample["trials"][0]["clip_a"]
        sample["trials"][1]["speaker_a"] = "different-person"
        with self.assertRaisesRegex(ValueError, "Source clip changes"):
            validate_trials(sample)

    def test_threshold_requires_distinct_voice_clips_not_only_many_pairs(self):
        # A seven-clip speaker can generate >=20 apparent genuine pairs,
        # but these are only seven underlying voice recordings.
        rows = []
        for i in range(7):
            for j in range(i + 1, 7):
                rows.append({
                    "id": f"genuine_{i}_{j}",
                    "session_a": f"session-{i}", "session_b": f"session-{j}",
                    "speaker_a": "person-A", "speaker_b": "person-A",
                    "clip_a": f"a{i}", "clip_b": f"a{j}",
                    "embedding_model": "camplus", "human_labeled": True,
                    "same_person": True, "cosine": 0.9,
                })
        for i in range(7):
            for j in range(3):
                rows.append({
                    "id": f"impostor_{i}_{j}",
                    "session_a": f"session-{i}", "session_b": f"other-{j}",
                    "speaker_a": "person-A", "speaker_b": "person-B",
                    "clip_a": f"a{i}", "clip_b": f"b{j}",
                    "embedding_model": "camplus", "human_labeled": True,
                    "same_person": False, "cosine": 0.1,
                })
        with self.assertRaisesRegex(ValueError, "distinct original voice clips"):
            validate_trials({"schema": SCHEMA, "trials": rows})

    def test_impostor_mislabeled_as_genuine_is_rejected(self):
        sample = trials("bad")
        sample["trials"][1]["same_person"] = True
        with self.assertRaisesRegex(ValueError, "Invalid trial"):
            validate_trials(sample)

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

"""Fixed195 triage tests: check weighting, error concentration and no cherry-picking."""
from __future__ import annotations

import unittest

from tools.fixed195_single_triage import group_summary, triage


def case(name, lang, ref, old=True):
    sample = {
        "id": name, "language": lang, "reference": ref, "kind": "single",
    }
    if old:
        sample["reused_fixed150_id"] = "fixed150-009"
    return sample


def result(sample, edits, predicted):
    return {
        "id": sample["id"], "language": sample["language"],
        "reference": sample["reference"], "hypothesis": "text",
        "character_edits": edits, "reference_characters": len(sample["reference"]),
        "predicted_language": predicted,
    }


class Fixed195TriageTests(unittest.TestCase):
    def test_character_weighted_and_split_old_new(self):
        samples = [
            case("fixed195-pt-s01", "pt", "A" * 100),
            case("fixed195-pt-s05", "pt", "A" * 10, False),
            case("fixed195-pt-s06", "pt", "A" * 20, False),
        ]
        results = {
            samples[0]["id"]: result(samples[0], 25, "es"),
            samples[1]["id"]: result(samples[1], 1, "pt"),
            samples[2]["id"]: result(samples[2], 4, "mul"),
        }
        summary = triage(samples, results, top=2, require_complete=False)
        pt = summary["per_language"]["pt"]
        self.assertAlmostEqual(pt["summary"]["cer_percent"], 23.077, places=3)
        self.assertEqual(pt["old"]["samples"], 1)
        self.assertEqual(pt["new"]["samples"], 2)
        self.assertEqual(pt["top_by_edits"][0]["id"], "fixed195-pt-s01")
        self.assertEqual(pt["top_by_cer"][0]["id"], "fixed195-pt-s01")
        self.assertEqual(pt["top_three_edit_share_percent"], 100.0)
        self.assertEqual(
            pt["detected_language_mismatch"],
            [{"id": "fixed195-pt-s01", "detected": "es"}],
        )
        self.assertEqual(
            pt["undetermined_or_multilingual_labels"],
            [{"id": "fixed195-pt-s06", "detected": "mul"}],
        )
        self.assertEqual(summary["completed"], 3)

    def test_missing_or_changed_reference_rejected(self):
        sample = case("fixed195-bn-s01", "bn", "Reference")
        bad = result(sample, 1, "bn")
        bad["reference"] = "Altered"
        with self.assertRaisesRegex(ValueError, "Reference text changed"):
            triage([sample], {sample["id"]: bad}, require_complete=False)

    def test_cannot_claim_full_completion_from_partial(self):
        sample = case("fixed195-en-s01", "en", "Hello")
        with self.assertRaisesRegex(ValueError, "Expected 130 singles"):
            triage([sample], {}, require_complete=True)

    def test_inference_failure_not_zero_cer(self):
        sample = case("fixed195-ru-s01", "ru", "Hello")
        report = triage([sample], {sample["id"]: {"error": "GPU Error"}},
                        require_complete=False)
        self.assertEqual(report["completed"], 0)
        self.assertEqual(len(report["failed"]), 1)
        self.assertIsNone(report["per_language"]["ru"]["summary"]["cer_percent"])

    def test_summary_no_reference_chars_is_unscored(self):
        self.assertIsNone(group_summary([])["cer_percent"])


if __name__ == "__main__":
    unittest.main()

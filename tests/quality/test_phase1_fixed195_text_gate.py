"""Fixed195 Phase 1 evaluation must validate all 130 human-reference cases."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase07.quality import score_pair
from tools.fixed195_content_review import LANGUAGE_ORDER
from tools.phase1_fixed195_text_gate import score_frozen_singles


def fixtures():
    cases, results = [], {}
    for i in range(130):
        lang = LANGUAGE_ORDER[i % len(LANGUAGE_ORDER)]
        case_id = f"fixed195-{lang}-s{i:03d}"
        reference = "hello everyone"
        case = {
            "id": case_id, "language": lang,
            "reference": reference,
        }
        cases.append(case)
        score = score_pair(reference, reference, language=lang)
        results[case_id] = {
            "reference": reference, "hypothesis": reference,
            "language": lang, "character_edits": score["character_edits"],
            "reference_characters": score["reference_characters"],
            "review_flags": [],
        }
    return cases, results


class Fixed195Phase1TextGateTests(unittest.TestCase):
    def run_fake(self, cases, results):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / "content_review_singles.json"
            path.write_text(json.dumps({
                "schema": "meeting-fixed195-content-review-v1",
                "fingerprint": "verified",
                "mode": "singles",
                "model": "large-v3",
                "single_results": results,
            }), encoding="utf-8")
            with (
                patch(
                    "tools.phase1_fixed195_text_gate.validate_singles",
                    return_value=cases,
                ),
                patch(
                    "tools.phase1_fixed195_text_gate.fingerprint",
                    return_value="verified",
                ),
            ):
                return score_frozen_singles(directory, path)

    def test_full_reference_singles_pass_without_fabricated_words(self):
        cases, results = fixtures()
        result = self.run_fake(cases, results)
        self.assertEqual(result["reference_case_count"], 130)
        self.assertTrue(result["pass"])
        self.assertEqual(result["overall"]["cer"], 0.0)
        self.assertEqual(result["overall"]["character_deletion_rate"], 0.0)
        self.assertNotIn("hello everyone", json.dumps(result))

    def test_one_missing_hypothesis_is_not_zero_error(self):
        cases, results = fixtures()
        first = cases[0]["id"]
        results[first]["hypothesis"] = ""
        original = score_pair(cases[0]["reference"], "", language=cases[0]["language"])
        results[first]["character_edits"] = original["character_edits"]
        result = self.run_fake(cases, results)
        self.assertFalse(result["pass"])
        self.assertEqual(result["overall"]["empty_predictions"], 1)
        self.assertGreater(result["overall"]["character_deletion_rate"], 0)

    def test_incomplete_batch_fails_closed(self):
        cases, results = fixtures()
        del results[cases[5]["id"]]
        with self.assertRaisesRegex(ValueError, "Missing, duplicated or extra"):
            self.run_fake(cases, results)

    def test_false_accuracy_from_modified_reference_rejected(self):
        cases, results = fixtures()
        results[cases[0]["id"]]["reference"] = "fabricated"
        with self.assertRaisesRegex(ValueError, "Unverifiable"):
            self.run_fake(cases, results)

    def test_changed_stored_cer_rejected(self):
        cases, results = fixtures()
        results[cases[0]["id"]]["character_edits"] = -1
        with self.assertRaisesRegex(ValueError, "Stored CER differs"):
            self.run_fake(cases, results)


if __name__ == "__main__":
    unittest.main()

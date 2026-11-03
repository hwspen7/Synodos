"""Regression tests for non-production Bengali/Portuguese Whisper A/B."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tools.fixed195_bn_pt_ablation import (
    case_score, override_window_lid, reference_case_check, summarize,
    verified_candidate, weighted,
)


class WhisperAblationContracts(unittest.TestCase):
    def test_weighted_cer_never_arithmetically_averages_clip_rates(self):
        row = weighted([
            {"edits": 20, "reference_chars": 100},
            {"edits": 0, "reference_chars": 10},
        ])
        self.assertAlmostEqual(row["cer"], 20 / 110)
        self.assertEqual(row["cer_percent"], 18.182)
        self.assertIsNone(weighted([])["cer"])

    def test_baseline_tampering_rejected(self):
        manifest = {"id": "fixed195-pt-s01", "language": "pt",
                    "reference": "Bom dia"}
        baseline = {"id": manifest["id"], "language": "pt",
                    "reference": "Bom dia", "hypothesis": "Bom dia",
                    "character_edits": 0, "reference_characters": 6}
        reference_case_check(manifest, baseline)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            reference_case_check(manifest, {**baseline, "character_edits": 1})
        with self.assertRaisesRegex(ValueError, "reference changed"):
            reference_case_check(manifest, {**baseline, "reference": "Wrong"})

    def test_explicit_language_is_not_automatic_lid(self):
        class Engine:
            def recognize(self, audio, language):
                class Pred:
                    language = "bn"
                    utterances = ()
                self.detected = os.environ.get("MEETING_WHISPER_WINDOW_LID")
                self.requested = language
                return Pred()
        engine = Engine()
        with patch.dict(os.environ, {"MEETING_WHISPER_WINDOW_LID": "1"}):
            candidate = verified_candidate(engine, [0.1], "bn", "whole_auto_verified")
            self.assertEqual(engine.requested, "auto")
            self.assertEqual(engine.detected, "0")
            self.assertEqual(os.environ["MEETING_WHISPER_WINDOW_LID"], "1")
            candidate = verified_candidate(engine, [0.1], "bn",
                                           "whole_forced_verified")
            self.assertEqual(engine.requested, "bn")
            self.assertEqual(os.environ["MEETING_WHISPER_WINDOW_LID"], "1")
        self.assertEqual(candidate["hypothesis"], "")
        with self.assertRaises(ValueError):
            verified_candidate(engine, [0.1], "bn", "unknown_variant")

    def test_candidate_reference_used_only_for_scoring(self):
        item = {"reference": "Bom dia", "language": "pt"}
        value = case_score(item, {"hypothesis": "Bom dia", "reported_language": "pt"})
        self.assertEqual(value["edits"], 0)
        value = case_score(item, {"hypothesis": "Bon dia", "reported_language": "pt"})
        self.assertEqual(value["edits"], 1)

    def test_summary_counts_regressions_and_does_not_hide_missing(self):
        def row(name, language, baseline_edits, other_edits):
            base = {"edits": baseline_edits, "reference_chars": 10,
                    "cer": baseline_edits / 10}
            candidate = ({"edits": other_edits, "reference_chars": 10,
                          "cer": other_edits / 10, "decode_ms": 100,
                          "invalid_replacement_chars": 0}
                         if other_edits is not None else None)
            return {"id": name, "language": language,
                    "baseline": base,
                    "candidates": {"whole_auto_verified": candidate}
                    if candidate is not None else {}}
        cases = [
            row("bn-1", "bn", 6, 3),
            row("bn-2", "bn", 1, 5),
            row("bn-3", "bn", 0, None),
            row("pt-1", "pt", 4, 0),
        ]
        score = summarize(cases, ("whole_auto_verified",))
        overall = score["overall"]
        self.assertEqual(overall["baseline"]["cer_percent"], 27.5)
        candidate = overall["candidates"]["whole_auto_verified"]
        self.assertEqual(candidate["improved_cases"], 2)
        self.assertEqual(candidate["worsened_cases"], 1)
        self.assertEqual(candidate["not_run"], 1)
        self.assertIsNone(candidate["delta_pp"])
        self.assertFalse(candidate["relative_cer_comparable"])
        self.assertFalse(score["deployment_approved"])
        self.assertIn("ORACLE", score["oracle_warning"].upper())


if __name__ == "__main__":
    unittest.main()

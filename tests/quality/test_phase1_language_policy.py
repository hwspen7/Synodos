"""Phase 1 supports only the ten accepted languages, not every model code."""
from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from speech.supported_languages import (
    PHASE1_LANGUAGE_CODES, PHASE1_LANGUAGE_SET,
    is_phase1_language, spoken_language, translation_target,
)
from speech.config import SpeechSettings
from tools.phase1_ten_language_gate import (
    evaluate_ten_language_selection, resolve_original_asr_results,
)


def archive():
    groups = {}
    for code in (*PHASE1_LANGUAGE_CODES, "pt", "bn"):
        n = 20 if code == "en" else 10
        chars = n * 100
        edits = 30 if code == "bn" else 20 if code == "pt" else 8
        groups[code] = {
            "samples": n, "reference_characters": chars,
            "character_edits": edits, "character_deletions": edits // 2,
            "empty_predictions": 0, "unscored_alignment": 0,
        }
    # The frozen archive has 130 single recordings: English 20,
    # and eleven other languages with 10 recordings each.
    return {
        "schema": "meeting-phase1-fixed195-text-gate-v1",
        "reference_case_count": 130,
        "per_language": groups,
        "model_and_code_fingerprint": "frozen-model",
        "asr_results_sha256": "frozen-samples",
    }


class Phase1LanguagePolicyTests(unittest.TestCase):
    def test_exact_ten_with_exclusions(self):
        self.assertEqual(PHASE1_LANGUAGE_CODES, (
            "en", "zh", "fr", "es", "ar", "id", "ja", "ru", "de", "hi",
        ))
        self.assertEqual(len(PHASE1_LANGUAGE_SET), 10)
        for removed in ("pt", "bn"):
            self.assertNotIn(removed, PHASE1_LANGUAGE_SET)

    def test_auto_detection_and_disabled_translation_are_legal(self):
        self.assertEqual(spoken_language("auto"), "auto")
        self.assertEqual(translation_target(""), "")
        self.assertEqual(translation_target(None), "")
        self.assertEqual(spoken_language("Chinese"), "zh")

    def test_removed_languages_rejected_even_when_provider_supports_them(self):
        for code in ("pt", "bn", "Portuguese", "Bengali", "ko", "fr-CA"):
            with self.subTest(language=code):
                with self.assertRaisesRegex(ValueError, "not enabled"):
                    spoken_language(code)
                with self.assertRaisesRegex(ValueError, "not enabled"):
                    translation_target(code)

    def test_invalid_or_unknown_is_not_silently_reinterpreted(self):
        for code in ("", "und", "mul", "autoo"):
            with self.subTest(language=code):
                with self.assertRaises(ValueError):
                    spoken_language(code)
        for code in ("auto", "und", "pt", "bn"):
            with self.assertRaises(ValueError):
                translation_target(code)
        self.assertTrue(is_phase1_language("Chinese"))
        self.assertFalse(is_phase1_language("bn"))

    def test_saved_meeting_api_rejects_unsupported_language_settings(self):
        from pydantic import ValidationError
        from api.meetings import MeetingSettingsPayload

        valid = MeetingSettingsPayload(language="auto", translate_to="")
        self.assertEqual(valid.language, "auto")
        self.assertEqual(valid.translate_to, "")
        alias = MeetingSettingsPayload(language="Chinese", translate_to="English")
        self.assertEqual(alias.language, "zh")
        self.assertEqual(alias.translate_to, "en")
        for key, language in (("language", "bn"), ("translate_to", "pt")):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                MeetingSettingsPayload(**{key: language})

    def test_intelligence_api_rejects_out_of_scope_new_translation(self):
        import asyncio
        from unittest.mock import patch
        from fastapi import HTTPException
        from api.intelligence import submit_job, JobRequest

        for language in ("pt", "bn", "ko"):
            with self.subTest(language=language), patch(
                "api.intelligence._enabled",
            ), self.assertRaises(HTTPException) as caught:
                asyncio.run(submit_job(JobRequest(
                    kind="live_translate", meeting_id="sample",
                    payload={"target": language, "segments": []},
                )))
            self.assertEqual(caught.exception.status_code, 400)

    def test_intelligence_worker_rejects_bypassed_out_of_scope_target(self):
        from unittest.mock import patch
        from intelligence.worker import process

        for kind in ("translate", "live_translate"):
            with self.subTest(kind=kind), patch(
                "intelligence.worker._context",
                return_value=(1, [{"id": 1, "text": "Hello"}]),
            ), self.assertRaisesRegex(ValueError, "not enabled"):
                process({
                    "kind": kind, "meeting_id": "sample",
                    "payload": {"target": "pt"},
                }, object())

    def test_invalid_backend_environment_fails_before_inference(self):
        SpeechSettings(language="auto", translate_to="").validate()
        for field, code in (("language", "pt"), ("translate_to", "bn")):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "not enabled"):
                    SpeechSettings(**{field: code}).validate()

    def test_frozen_130_archive_is_not_mutated_to_select_110(self):
        original = archive()
        result = evaluate_ten_language_selection(original)
        self.assertEqual(result["selected_reference_case_count"], 110)
        self.assertEqual(result["archive_reference_case_count"], 130)
        self.assertEqual(set(result["per_language"]), PHASE1_LANGUAGE_SET)
        self.assertEqual(result["excluded_from_phase1"], ["bn", "pt"])
        self.assertTrue(result["pass"])
        self.assertEqual(len(original["per_language"]), 12)
        self.assertNotIn("pass", original)

    def test_bad_supported_language_still_fails_target_selection(self):
        original = archive()
        original["per_language"]["hi"]["character_edits"] = 300
        result = evaluate_ten_language_selection(original)
        self.assertFalse(result["pass"])
        self.assertFalse(result["per_language"]["hi"]["pass"])

    def test_latest_summary_resolves_only_its_exact_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "content_review_singles_20261104-160000.json"
            summary = root / "content_review_singles_20261104-160000_summary.json"
            raw.write_text(json.dumps({
                "schema": "meeting-fixed195-content-review-v1",
                "mode": "singles",
                "single_results": {"fixture": {}},
            }), encoding="utf-8")
            summary.write_text(json.dumps({
                "schema": "meeting-fixed195-content-review-v1-summary",
                "mode": "singles",
                "singles": {"overall": {"completed": 130}},
            }), encoding="utf-8")
            self.assertEqual(resolve_original_asr_results(raw), raw)
            self.assertEqual(resolve_original_asr_results(summary), raw)
            self.assertEqual(json.loads(raw.read_text(encoding="utf-8"))[
                "single_results"], {"fixture": {}})

    def test_summary_without_original_is_rejected_without_fabricating_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = root / "content_review_singles_20261104_summary.json"
            summary.write_text(json.dumps({
                "schema": "meeting-fixed195-content-review-v1-summary",
                "mode": "singles",
            }), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "matching original"):
                resolve_original_asr_results(summary)
            raw = root / "content_review_singles_20261104.json"
            raw.write_text(json.dumps({
                "schema": "meeting-fixed195-content-review-v1-summary",
                "mode": "singles",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a complete"):
                resolve_original_asr_results(summary)

    def test_unrelated_or_tampered_summary_is_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = Path(folder) / "content_review_singles_summary.json"
            summary.write_text(json.dumps({
                "schema": "meeting-phase1-ten-language-text-gate-v1",
                "mode": "singles",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a valid"):
                resolve_original_asr_results(summary)

    def test_incomplete_source_archive_cannot_masquerade_as_ten_language(self):
        original = archive()
        del original["per_language"]["bn"]
        with self.assertRaises(ValueError):
            evaluate_ten_language_selection(original)


if __name__ == "__main__":
    unittest.main()

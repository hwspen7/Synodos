"""Deterministic, no GPU/download/DB ASR selection contract tests."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.asr_model_selection import (
    PREDICTIONS_SCHEMA, SCORE_SCHEMA, compare_reports, distance, normalize,
    p95, score_predictions,
)


class ASRModelSelectionTests(unittest.TestCase):
    def test_text_normalization_and_distance(self):
        self.assertEqual(normalize("Hello, WORLD!")[0], ["hello", "world"])
        self.assertEqual(normalize("\u4f60\u597d\uff01")[1], list("\u4f60\u597d"))
        self.assertEqual(normalize("\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e")[0],
                         ["\u0928\u092e\u0938\u094d\u0924\u0947", "\u0926\u0941\u0928\u093f\u092f\u093e"])
        self.assertEqual(distance(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(p95([1, 2, 3, 4, 5]), 5)

    def test_score_is_model_independent_and_fail_closed(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = root / "manifest.json"
            prediction_path = root / "pred.json"
            from tools.freeze_fleurs_asr_corpus import LANGUAGES
            cases = [
                {"id": "id_" + lang, "language": lang,
                 "audio_sha256": "a" * 64,
                 "reference_text": "hello world", "audio_duration_seconds": 2}
                for lang in LANGUAGES
            ]
            manifest = {"per_language": 1, "benchmark_role": "development", "cases": cases}
            observations = [
                {"id": x["id"], "audio_sha256": "a" * 64,
                 "text": "hello world", "error": "",
                 "latency_seconds": .2, "detected_language": x["language"]}
                for x in cases
            ]
            pred = {
                "schema": PREDICTIONS_SCHEMA, "model": "whisper-large-v3",
                "model_id": "large-v3", "manifest_sha256": "b" * 64,
                "benchmark_role": "development", "language_mode": "auto",
                "reference_transcript_used_during_inference": False,
                "production_asr_modified": False, "observations": observations,
            }
            prediction_path.write_text(json.dumps(pred))
            with patch("tools.asr_model_selection.load_corpus",
                       return_value=(manifest, "b" * 64)), patch(
                "tools.asr_model_selection.require_e_drive", side_effect=lambda p: p
            ):
                with patch("tools.asr_model_selection.file_hash", return_value="c" * 64):
                    report = score_predictions(manifest_path, prediction_path)
                self.assertEqual(report["macro_primary_error_rate"], 0)
                self.assertFalse(report["production_release_pass"])
                self.assertEqual(report["evaluated_cases"], 10)
                pred["observations"][0]["id"] = "tampered"
                prediction_path.write_text(json.dumps(pred))
                with self.assertRaises(ValueError):
                    score_predictions(manifest_path, prediction_path)

    def test_cross_corpus_ranking_is_rejected(self):
        with TemporaryDirectory() as td:
            a, b = Path(td) / "a.json", Path(td) / "b.json"
            from tools.freeze_fleurs_asr_corpus import LANGUAGES
            report = {
                "schema": SCORE_SCHEMA, "source_manifest_sha256": "a" * 64,
                "language_mode": "auto", "benchmark_role": "sealed_test",
                "evaluated_cases": 500, "per_language_case_count": 50,
                "per_language": {lang: {"cases": 50} for lang in LANGUAGES},
                "live_meeting_gold_evaluated": False,
                "production_release_pass": False, "model": "whisper-large-v3",
                "eligible_for_read_speech_ranking": True,
                "macro_primary_error_rate": .1, "worst_language_error_rate": .3,
                "empty_prediction_count": 0, "failed_case_count": 0,
                "source_predictions_sha256": "c" * 64,
            }
            a.write_text(json.dumps(report))
            report["source_manifest_sha256"] = "b" * 64
            report["model"] = "qwen3-asr-1.7b"
            b.write_text(json.dumps(report))
            with patch("tools.asr_model_selection.require_e_drive", side_effect=lambda p: p):
                with self.assertRaises(ValueError):
                    compare_reports([a, b])

    def test_compatible_pair_ranks_without_production_approval(self):
        with TemporaryDirectory() as td:
            from tools.freeze_fleurs_asr_corpus import LANGUAGES
            paths = []
            for i, model in enumerate(("whisper-large-v3", "qwen3-asr-1.7b")):
                row = {
                    "schema": SCORE_SCHEMA, "source_manifest_sha256": "a" * 64,
                    "language_mode": "auto", "benchmark_role": "development",
                    "evaluated_cases": 200, "per_language_case_count": 20,
                    "per_language": {lang: {} for lang in LANGUAGES},
                    "live_meeting_gold_evaluated": False,
                    "production_release_pass": False, "model": model,
                    "eligible_for_read_speech_ranking": True,
                    "macro_primary_error_rate": .2 - i * .05,
                    "worst_language_error_rate": .4,
                    "empty_prediction_count": i,
                    "failed_case_count": 0,
                    "source_predictions_sha256": "a" * 64,
                }
                path = Path(td) / (str(i) + ".json")
                path.write_text(json.dumps(row))
                paths.append(path)
            with patch("tools.asr_model_selection.require_e_drive", side_effect=lambda p: p):
                result = compare_reports(paths)
            self.assertEqual(result["best_on_this_published_read_speech_cohort"], "qwen3-asr-1.7b")
            self.assertIsNone(result["best_production_meeting_model"])
            self.assertFalse(result["may_auto_promote"])


if __name__ == "__main__":
    unittest.main()

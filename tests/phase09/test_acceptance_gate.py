"""No simulated transcript can certify multilingual acoustic performance."""
import unittest

from tools.phase09_acceptance import evaluate


def report(infer=True, error=0, language="en"):
    return {
        "infer": infer,
        "engines": {
            "whisper": {
                "samples": [{
                    "index": n, "expected_language": language,
                    "predicted_language": language,
                    "reference_characters": 20, "character_edits": error,
                    "reference_tokens": 5, "token_edits": 0,
                    "latency_ms": 500,
                } for n in range(5)],
            }
        },
    }


class AcceptanceGateTests(unittest.TestCase):
    def test_real_annotated_pass(self):
        result = evaluate(report(), ["en"])
        self.assertTrue(result["whisper"]["pass"])

    def test_simulated_report_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(report(infer=False), ["en"])

    def test_missing_multilingual_coverage_fails_closed(self):
        result = evaluate(report(), ["en", "ja"])
        self.assertFalse(result["whisper"]["pass"])
        self.assertTrue(any("ja:" in error for error in result["whisper"]["failures"]))

    def test_high_character_errors_fail(self):
        result = evaluate(report(error=10), ["en"])
        self.assertFalse(result["whisper"]["pass"])

    def test_language_identification_failure_fails(self):
        bad = report()
        bad["engines"]["whisper"]["samples"][0]["predicted_language"] = "fr"
        self.assertFalse(evaluate(bad, ["en"])["whisper"]["pass"])

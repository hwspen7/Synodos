"""Regression coverage for immutable Fixed10 manifest and evidence-only scores."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.fixed10_prepare import build
from tools.fixed10_dashboard import content_recall, prepare


def _cases():
    singles = [{
        "id": f"fixed10-{i:02d}", "audio": f"audio/{i}.wav",
        "kind": "single", "language": lang, "duration_s": 2.0,
    } for i, lang in enumerate(["en", "zh", "es", "fr", "de", "ko", "pt", "ru"], 1)]
    overlaps = [{
        "id": f"fixed10-{i:02d}", "audio": f"audio/{i}.wav",
        "kind": "overlap", "language": "mul", "duration_s": 20.0,
        "speakers": [{
            "speaker": f"S{j}", "language": "fr", "reference": "Bonjour merci",
            "start_ms": j * 1000, "end_ms": (j + 8) * 1000,
        } for j in range(1, 4)],
    } for i in (9, 10)]
    return singles + overlaps


class Fixed10SmokeTests(unittest.TestCase):
    def test_not_run_is_not_a_pass(self):
        out = prepare({"cases": _cases()}, None, None)
        self.assertFalse(out["real_inference"])
        self.assertFalse(out["release_certified"])
        self.assertEqual(out["cases"], 10)
        self.assertEqual(out["gates"][0]["status"], "NOT RUN")

    def test_unverified_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare({"cases": _cases()}, {"infer": False, "samples": 8}, None)

    def test_failed_real_inference_is_not_certified(self):
        out = prepare({"cases": _cases()}, {
            "infer": True, "samples": 8,
            "engines": {"whisper": {"samples": [
                {"index": i, "error": "RuntimeError"} for i in range(8)
            ]}},
        }, None)
        self.assertEqual(out["gates"][0]["status"], "BLOCKED")

    def test_latency_failure_is_reported(self):
        rows = [{
            "index": i, "reference_characters": 100,
            "character_edits": 0, "reference_tokens": 10,
            "token_edits": 0, "latency_ms": 4001,
            "expected_language": lang, "predicted_language": lang,
            "cer": 0, "mixed_token_error_rate": 0,
        } for i, lang in enumerate(["en", "zh", "es", "fr", "de", "ko", "pt", "ru"])]
        out = prepare({"cases": _cases()}, {
            "infer": True, "samples": 8, "engines": {"whisper": {"samples": rows}},
        }, None)
        self.assertEqual(out["gates"][0]["status"], "REVIEW")
        self.assertAlmostEqual(out["engines"]["whisper"]["latency_p95_ms"], 4001)

    def test_content_word_recall_is_multiset_based(self):
        self.assertAlmostEqual(content_recall("hello hello testing", "hello testing"), 2/3)
        self.assertIsNone(content_recall("a b", "a b"))

    def test_lock_fails_when_audio_file_is_modified(self):
        with TemporaryDirectory() as dirname:
            root = Path(dirname)
            cases = _cases()
            for case in cases:
                path = root / case["audio"]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"audio")
                case["sha256"] = sha256(b"audio").hexdigest()
            (root / "lock.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
            (root / "cases.json").write_text("{}", encoding="utf-8")
            (root / "manifest_single.jsonl").write_text("{}\n", encoding="utf-8")
            build(root, verify_only=True)
            (root / cases[1]["audio"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                build(root, verify_only=True)

    def test_existing_lock_never_redownloads(self):
        with TemporaryDirectory() as dirname:
            root = Path(dirname)
            cases = _cases()
            (root / "lock.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build(root, verify_only=False)


if __name__ == "__main__":
    unittest.main()

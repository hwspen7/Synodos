"""Fixed180 allocation and backwards-compatible source provenance contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.fixed180_prepare import (
    LANGUAGES, REUSED_FIXED150_IDS, FLEURS_VALIDATION_ROWS,
    SINGLES_PER_LANGUAGE, MULTI_PER_LANGUAGE, SCHEMA,
    create_outputs, verify_case,
)


class Fixed180Contracts(unittest.TestCase):
    def test_exact_twelve_languages_4_old_6_new_and_5_real_dialogues(self):
        self.assertEqual(len(LANGUAGES), 12)
        self.assertEqual(set(REUSED_FIXED150_IDS), {lang for lang, _ in LANGUAGES})
        self.assertEqual(SINGLES_PER_LANGUAGE, 10)
        self.assertEqual(MULTI_PER_LANGUAGE, 5)
        self.assertEqual(len(FLEURS_VALIDATION_ROWS), 6)
        self.assertEqual(len(set(FLEURS_VALIDATION_ROWS)), 6)
        self.assertTrue(set(FLEURS_VALIDATION_ROWS).isdisjoint({19, 43, 77, 101}))
        ids = [cid for values in REUSED_FIXED150_IDS.values() for cid in values]
        self.assertEqual(len(ids), 48)
        self.assertEqual(len(set(ids)), 48)
        self.assertEqual(
            REUSED_FIXED150_IDS["en"],
            ("fixed150-009", "fixed150-013",
             "fixed150-017", "fixed150-018"),
        )
        self.assertEqual(REUSED_FIXED150_IDS["pt"][-1], "fixed150-048")
        self.assertEqual(REUSED_FIXED150_IDS["bn"][0], "fixed150-089")

    def test_incomplete_subset_never_gets_final_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {
                "revisions": {}, "cases": {
                    "fixed180-en-s01": {
                        "id": "fixed180-en-s01",
                        "audio": "missing.wav", "sha256": "hash",
                        "language": "en", "kind": "single",
                    }
                },
            }
            with patch("tools.fixed180_prepare.verify_case"):
                self.assertFalse(create_outputs(root, state))
            self.assertFalse((root / "lock.json").exists())
            self.assertEqual(
                len((root / "manifest_single.jsonl").read_text().splitlines()), 1
            )

    def test_complete_180_cases_are_frozen_with_each_language_10_plus_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {}
            for lang, _ in LANGUAGES:
                for idx in range(1, 11):
                    cid = f"fixed180-{lang}-s{idx:02d}"
                    case = {
                        "id": cid, "audio": cid + ".wav", "kind": "single",
                        "language": lang, "sha256": cid,
                        "reused_fixed150_id": (
                            REUSED_FIXED150_IDS[lang][idx-1] if idx <= 4 else None
                        ),
                    }
                    cases[cid] = case
                for idx in range(1, 6):
                    cid = f"fixed180-{lang}-m{idx:02d}"
                    cases[cid] = {
                        "id": cid, "audio": cid + ".wav", "kind": "multi",
                        "language": lang, "sha256": cid,
                        "reference_segments": [
                            {"speaker": "speaker_1"},
                            {"speaker": "speaker_2"},
                        ],
                    }
            with patch("tools.fixed180_prepare.verify_case"):
                self.assertTrue(create_outputs(root, {
                    "revisions": {"google/fleurs": "mock-pinned"},
                    "cases": cases,
                }))
            lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
            self.assertEqual(len(lock["cases"]), 180)
            self.assertEqual(lock["reused_fixed150_singles"], 48)
            self.assertEqual(lock["new_fleurs_singles"], 72)
            self.assertEqual(lock["real_two_speaker_conversations"], 60)
            self.assertEqual(lock["schema"], SCHEMA)
            self.assertEqual(
                len((root / "manifest_single.jsonl").read_text().splitlines()), 120
            )
            self.assertEqual(
                len((root / "manifest_multi.jsonl").read_text().splitlines()), 60
            )
            for lang, _ in LANGUAGES:
                selected = [x for x in lock["cases"] if x["language"] == lang]
                self.assertEqual(len(selected), 15)
                self.assertEqual(sum(x["kind"] == "single" for x in selected), 10)
                self.assertEqual(sum(x["kind"] == "multi" for x in selected), 5)


if __name__ == "__main__":
    unittest.main()

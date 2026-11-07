"""Fixed195 allocation and backwards-compatible source provenance contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evaluation.corpora.fixed195_prepare import (
    LANGUAGES, REUSED_FIXED150_IDS, FLEURS_VALIDATION_ROWS,
    SINGLES_PER_LANGUAGE, MULTI_PER_LANGUAGE, SCHEMA,
    ENGLISH_SINGLES, ENGLISH_MULTI, ENGLISH_FLEURS_ROWS,
    singles_for, multi_for, rows_for, choose_dialogues, collect_segments,
    create_outputs, verify_case,
)


class Fixed195Contracts(unittest.TestCase):
    def test_exact_twelve_languages_with_twice_the_english_coverage(self):
        self.assertEqual(len(LANGUAGES), 12)
        self.assertEqual(set(REUSED_FIXED150_IDS), {lang for lang, _ in LANGUAGES})
        self.assertEqual(SINGLES_PER_LANGUAGE, 10)
        self.assertEqual(MULTI_PER_LANGUAGE, 5)
        self.assertEqual(len(FLEURS_VALIDATION_ROWS), 6)
        self.assertEqual(len(ENGLISH_FLEURS_ROWS), 8)
        self.assertEqual(len(set(ENGLISH_FLEURS_ROWS)), 8)
        self.assertEqual(ENGLISH_SINGLES, 20)
        self.assertEqual(ENGLISH_MULTI, 10)
        self.assertEqual(singles_for("en"), 20)
        self.assertEqual(multi_for("en"), 10)
        self.assertEqual(singles_for("pt"), 10)
        self.assertEqual(multi_for("pt"), 5)
        self.assertEqual(rows_for("en"), ENGLISH_FLEURS_ROWS)
        self.assertEqual(rows_for("pt"), FLEURS_VALIDATION_ROWS)
        self.assertEqual(len(set(FLEURS_VALIDATION_ROWS)), 6)
        self.assertTrue(set(FLEURS_VALIDATION_ROWS).isdisjoint({19, 43, 77, 101}))
        ids = [cid for values in REUSED_FIXED150_IDS.values() for cid in values]
        self.assertEqual(len(ids), 56)
        self.assertEqual(len(set(ids)), 56)
        self.assertEqual(
            REUSED_FIXED150_IDS["en"],
            tuple(f"fixed150-{i:03d}" for i in range(9, 21)),
        )
        self.assertEqual(REUSED_FIXED150_IDS["pt"][-1], "fixed150-048")
        self.assertEqual(REUSED_FIXED150_IDS["bn"][0], "fixed150-089")

    def test_incomplete_subset_never_gets_final_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {
                "revisions": {}, "cases": {
                    "fixed195-en-s01": {
                        "id": "fixed195-en-s01",
                        "audio": "missing.wav", "sha256": "hash",
                        "language": "en", "kind": "single",
                    }
                },
            }
            with patch("tools.evaluation.corpora.fixed195_prepare.verify_case"):
                self.assertFalse(create_outputs(root, state))
            self.assertFalse((root / "lock.json").exists())
            self.assertEqual(
                len((root / "manifest_single.jsonl").read_text().splitlines()), 1
            )

    def test_complete_195_cases_are_frozen_with_english_20_plus_10(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {}
            for lang, _ in LANGUAGES:
                for idx in range(1, singles_for(lang) + 1):
                    cid = f"fixed195-{lang}-s{idx:02d}"
                    case = {
                        "id": cid, "audio": cid + ".wav", "kind": "single",
                        "language": lang, "sha256": cid,
                        "reused_fixed150_id": (
                            REUSED_FIXED150_IDS[lang][idx-1]
                            if idx <= len(REUSED_FIXED150_IDS[lang]) else None
                        ),
                    }
                    cases[cid] = case
                for idx in range(1, multi_for(lang) + 1):
                    cid = f"fixed195-{lang}-m{idx:02d}"
                    cases[cid] = {
                        "id": cid, "audio": cid + ".wav", "kind": "multi",
                        "language": lang, "sha256": cid,
                        "reference_segments": [
                            {"speaker": "speaker_1"},
                            {"speaker": "speaker_2"},
                        ],
                    }
            with patch("tools.evaluation.corpora.fixed195_prepare.verify_case"):
                self.assertTrue(create_outputs(root, {
                    "revisions": {"google/fleurs": "mock-pinned"},
                    "cases": cases,
                }))
            lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
            self.assertEqual(len(lock["cases"]), 195)
            self.assertEqual(lock["reused_fixed150_singles"], 56)
            self.assertEqual(lock["new_fleurs_singles"], 74)
            self.assertEqual(lock["real_two_speaker_conversations"], 65)
            self.assertEqual(lock["schema"], SCHEMA)
            self.assertEqual(
                len((root / "manifest_single.jsonl").read_text().splitlines()), 130
            )
            self.assertEqual(
                len((root / "manifest_multi.jsonl").read_text().splitlines()), 65
            )
            for lang, _ in LANGUAGES:
                selected = [x for x in lock["cases"] if x["language"] == lang]
                self.assertEqual(len(selected), singles_for(lang) + multi_for(lang))
                self.assertEqual(sum(x["kind"] == "single" for x in selected),
                                 singles_for(lang))
                self.assertEqual(sum(x["kind"] == "multi" for x in selected),
                                 multi_for(lang))


if __name__ == "__main__":
    unittest.main()

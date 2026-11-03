"""Contract tests for the twelve-language, 60 single + 60 real-dialogue benchmark."""
from __future__ import annotations

import unittest

from tools.fixed120_prepare import (
    LANGUAGES, FLEURS_VALIDATION_ROWS, collect_segments,
    choose_dialogues, SCHEMA,
)
from speech.language import normalize_language
from speech.qwen import qwen_language_name


def sample(sid: str, language: str, overlap: float) -> dict:
    return {
        "sample_id": sid, "language": language,
        "n_speakers": 2, "duration_s": 67.0,
        "speaker1_id": f"a-{sid}", "speaker2_id": f"b-{sid}",
        "clip_overlap_frac": overlap,
        "word_alignments": [
            {"speaker": speaker, "start": start, "end": start + 2.0,
             "timing_source": None, "words": [
                 {"start": start + i*.3, "end": start + i*.3+.15,
                  "text": f"word{i}"} for i in range(4)
             ]}
            for speaker, start in ((1, 0.0), (2, 3.0))
        ],
    }


class Fixed120Contracts(unittest.TestCase):
    def test_bengali_names_are_recognized_without_false_qwen_support(self):
        self.assertEqual(normalize_language("Bengali"), "bn")
        self.assertEqual(normalize_language("Bangla"), "bn")
        # Qwen3-ASR's documented set excludes Bengali; Whisper is primary.
        with self.assertRaises(ValueError):
            qwen_language_name("bn")

    def test_12_languages_five_new_fleurs_validation_rows(self):
        self.assertEqual(len(LANGUAGES), 12)
        self.assertEqual(len(set(k for k, _ in LANGUAGES)), 12)
        self.assertEqual(len(set(k for _, k in LANGUAGES)), 12)
        self.assertEqual(len(FLEURS_VALIDATION_ROWS), 5)
        self.assertTrue(set(FLEURS_VALIDATION_ROWS).isdisjoint(
            {19, 43, 77, 101}
        ))
        self.assertTrue(all(x < 180 for x in FLEURS_VALIDATION_ROWS))

    def test_real_dialogue_selection_is_deterministic_and_spread_by_overlap(self):
        rows = [sample(f"clip-{i:02d}", "bn", i / 12) for i in range(10)]
        rows = rows[::-1]
        chosen = choose_dialogues(rows, "bn")
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len({x["sample_id"] for x in chosen}), 5)
        overlaps = [x["clip_overlap_frac"] for x in chosen]
        self.assertEqual(overlaps, sorted(overlaps))
        self.assertLess(overlaps[0], .2)
        self.assertGreater(overlaps[-1], .55)
        self.assertEqual(
            [x["sample_id"] for x in chosen],
            [x["sample_id"] for x in choose_dialogues(rows[::-1], "bn")]
        )

    def test_rejects_single_speaker_and_missing_second_speaker(self):
        rows = [sample(str(i), "ar", .15) for i in range(5)]
        rows[0]["n_speakers"] = 1
        with self.assertRaisesRegex(ValueError, "valid two-speaker"):
            choose_dialogues(rows, "ar")
        rows[0]["n_speakers"] = 2
        rows[1]["word_alignments"] = rows[1]["word_alignments"][:1]
        with self.assertRaises(ValueError):
            choose_dialogues(rows, "ar")

    def test_no_synthetic_or_fabricated_transcript_timing(self):
        row = sample("clip", "pt", .05)
        rows = collect_segments(row["word_alignments"])
        self.assertEqual(len(rows), 2)
        self.assertEqual({x["speaker"] for x in rows},
                         {"speaker_1", "speaker_2"})
        self.assertEqual(rows[0]["start_ms"], 0)
        self.assertEqual(rows[1]["start_ms"], 3000)
        row["word_alignments"][0]["timing_source"] = "pseudo"
        rows = collect_segments(row["word_alignments"])
        self.assertEqual(rows[0]["timing_source"], "pseudo")
        self.assertEqual(SCHEMA, "meeting-fixed120-real-v1")


if __name__ == "__main__":
    unittest.main()

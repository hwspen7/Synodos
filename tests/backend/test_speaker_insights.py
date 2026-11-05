"""Regression checks for the scoped per-speaker AI contribution feature."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from intelligence.speaker_insights import (
    analyze_speaker, choose_speaker_turns, safe_speaker_insights,
    speaker_prompt, split_speaker_turns,
)


def turn(number, speaker, text, *, name=None, start=None):
    start = number * 2000 if start is None else start
    return {
        "id": number, "speaker_id": speaker, "speaker": name or f"Speaker {speaker}",
        "start_ms": start, "end_ms": start + 1100, "text": text,
    }


class FakeGateway:
    def __init__(self):
        self.prompts = []

    def model_for(self, mode):
        assert mode == "meeting"
        return "test-model"

    def structured(self, system, prompt, model):
        self.prompts.append(prompt)
        ids = [int(line.split("]")[0][1:])
               for line in prompt.splitlines() if line.startswith("[")]
        assert ids
        return {
            "summary": "Speaker describes a planning task.",
            "summary_source_ids": [ids[0], 99_999],
            "viewpoints": [{"title": "Concern", "source_ids": [ids[0], 99_999]}],
            "decisions": [{"title": "Unapproved idea", "source_ids": [99_999]}],
            "actions": [{"title": "Draft outline", "source_ids": [ids[0]],
                         "owner": "Unknown", "due_date": "next week"}],
            "questions": [{"title": "Timeline?", "source_ids": [ids[0]]}],
        }


class SpeakerInsightsTest(unittest.TestCase):
    def test_selects_only_exact_speaker_in_order(self):
        turns = [turn(2, 2, "NOT FOR ANALYSIS"),
                 turn(3, 1, "Second"), turn(1, 1, "First")]
        name, selected = choose_speaker_turns(turns, 1)
        self.assertEqual(name, "Speaker 1")
        self.assertEqual([s["id"] for s in selected], [1, 3])

    def test_unidentified_or_missing_speaker_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_speaker_turns([turn(1, -1, "hi")], -1)
        with self.assertRaises(ValueError):
            choose_speaker_turns([turn(1, 1, "hi")], 9)
        with self.assertRaises(ValueError):
            choose_speaker_turns([turn(1, 1, "hi")], True)

    def test_ambiguous_labels_rejected(self):
        turns = [turn(1, 1, "A", name="Speaker 1"),
                 turn(2, 1, "B", name="Other person")]
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            choose_speaker_turns(turns, 1)

    def test_citations_never_point_to_another_speaker(self):
        raw = {"summary": "Summary", "summary_source_ids": [1, 2],
               "viewpoints": [{"title": "Point", "source_ids": [2]},
                              {"title": "Supported", "source_ids": [1, 2]}],
               "decisions": [{"title": "Unsupported", "source_ids": [2]}]}
        result = safe_speaker_insights(raw, 1, "Speaker 1", [turn(1, 1, "A")])
        self.assertEqual(result["summary_source_ids"], [1])
        self.assertEqual(result["viewpoints"][0]["source_ids"], [1])
        self.assertEqual(len(result["viewpoints"]), 1)
        self.assertEqual(result["decisions"], [])

    def test_uncited_summary_discarded(self):
        result = safe_speaker_insights(
            {"summary": "Made-up claim", "summary_source_ids": [200]},
            1, "Speaker 1", [turn(1, 1, "A")],
        )
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["summary_source_ids"], [])

    def test_scoped_model_prompt_and_strict_validation(self):
        gateway = FakeGateway()
        result = analyze_speaker(
            [turn(1, 1, "First statement"),
             turn(2, 2, "SECRET OTHER SPEAKER"),
             turn(3, 1, "Second statement")],
            1, gateway,
        )
        self.assertEqual(result["speaker_id"], 1)
        self.assertEqual(result["segment_ids"], [1, 3])
        self.assertTrue(result["human_review_required"])
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["actions"][0]["due_date"], None)
        self.assertTrue(all("SECRET OTHER SPEAKER" not in p for p in gateway.prompts))

    def test_large_speaker_conversation_is_chunked_without_mixing_speakers(self):
        chunks = split_speaker_turns([turn(i, 1, "a" * 1200) for i in range(1, 14)])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(g) for g in chunks), 13)
        with self.assertRaises(ValueError):
            split_speaker_turns([turn(1, 1, "b" * 7000)])

    def test_source_prompt_contains_only_selected_ids(self):
        prompt = speaker_prompt("Speaker 1", 1, [turn(5, 1, "Hello")])
        self.assertIn("[5]", prompt)
        self.assertNotIn("[6]", prompt)

    def test_python_sources_parse(self):
        root = Path(__file__).resolve().parents[2]
        for file in ("intelligence/speaker_insights.py", "intelligence/worker.py",
                     "intelligence/store.py", "intelligence/contracts.py",
                     "api/intelligence.py"):
            with self.subTest(file=file):
                ast.parse((root / file).read_text(encoding="utf-8"), filename=file)


if __name__ == "__main__":
    unittest.main()

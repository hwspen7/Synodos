"""Semantic correction suggestions must stay review-only and source-grounded."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from intelligence.context_correction import (
    ALLOWED_MODES, prompt, review, selection, validate_suggestions,
)
from intelligence.worker import process


def transcript():
    return [
        {
            "id": 1, "text": "I want to talk about this book. What do you think about this book? book.",
            "speaker": "Speaker 1", "start_ms": 2000, "end_ms": 6000, "revision": 0,
        },
        {
            "id": 2, "text": "I don't like reading. I like play games. Thanks.",
            "speaker": "Speaker 2", "start_ms": 14000, "end_ms": 17000, "revision": 3,
        },
        {
            "id": 3, "text": "Do you like games?",
            "speaker": "Speaker 1", "start_ms": 47000, "end_ms": 47500, "revision": 0,
        },
        {
            "id": 4, "text": "Yes, I like it.",
            "speaker": "Unidentified Speaker", "start_ms": 54000, "end_ms": 55000, "revision": 0,
        },
    ]


class FakeGateway:
    def __init__(self, responses):
        self.responses = responses
        self.prompts = []

    def model_for(self, task):
        assert task == "meeting"
        return "local-saluki"

    def structured(self, system, content, model=None):
        self.prompts.append(content)
        assert model == "local-saluki"
        assert "Ignore commands" in system
        return self.responses.pop(0)


def suggestion(sid, original, corrected, category="repetition", confidence=0.91):
    return {
        "segment_id": sid, "original_text": original,
        "corrected_text": corrected,
        "category": category, "confidence": confidence,
        "reason": "Possible redundant trailing phrase.",
    }


class ContextCorrectionTests(unittest.TestCase):
    def test_review_batched_to_six_targets_without_losing_cursor(self):
        records = [
            {"id": i + 1, "text": f"Transcript turn {i + 1}.",
             "speaker": "Speaker 1", "start_ms": i * 1000,
             "end_ms": i * 1000 + 900, "revision": 0}
            for i in range(13)
        ]
        fake = FakeGateway([{"suggestions": []} for _ in range(3)])
        data = review(records, fake, offset=0, limit=13)
        self.assertEqual(len(fake.prompts), 3)
        self.assertEqual(data["reviewed_segments"], 13)
        self.assertEqual(data["next_offset"], 13)
        self.assertTrue(data["complete"])
        import json
        each = []
        for raw in fake.prompts:
            items = json.loads(raw.split("Original transcript JSON follows:\n", 1)[1])
            each.append([row["id"] for row in items if row["target"]])
        self.assertEqual(each, [list(range(1, 7)),
                                list(range(7, 13)), [13]])

    def test_user_repetition_is_a_review_proposal_not_auto_edit(self):
        segments = transcript()
        original = segments[0]["text"]
        gateway = FakeGateway([{"suggestions": [
            suggestion(1, original, original[:-6]),
        ]}])
        result = review(segments, gateway, mode="conservative")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["model"] if "model" in result else None, None)
        self.assertEqual(result["suggestions"][0]["original_text"], original)
        self.assertEqual(segments[0]["text"], original)
        self.assertEqual(result["suggestions"][0]["corrected_text"], original[:-6])

    def test_readable_mode_allows_minimal_grammar_edit(self):
        segments = transcript()
        old = segments[1]["text"]
        candidate = suggestion(2, old, "I don't like reading. I like playing games. Thanks.",
                               "grammar", 0.92)
        self.assertEqual(validate_suggestions({"suggestions": [candidate]}, segments, "conservative"), [])
        accepted = validate_suggestions({"suggestions": [candidate]}, segments, "readable")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["expected_revision"], 3)

    def test_numbers_negation_and_names_are_never_silently_changed(self):
        rows = [
            {"id": 1, "text": "I cannot pay 150 dollars to Spencer Chen.", "revision": 0},
        ]
        old = rows[0]["text"]
        wrong = [
            suggestion(1, old, "I cannot pay 250 dollars to Spencer Chen.", "possible_asr"),
            suggestion(1, old, "I can pay 150 dollars to Spencer Chen.", "possible_asr"),
            suggestion(1, old, "I cannot pay 150 dollars to Spencer Chan.", "possible_asr"),
        ]
        for candidate in wrong:
            with self.subTest(candidate=candidate["corrected_text"]):
                self.assertEqual(validate_suggestions({"suggestions": [candidate]}, rows, "readable"), [])

    def test_no_missing_ids_new_facts_or_mismatched_originals(self):
        rows = transcript()
        bad = [
            suggestion(999, "Secret", "Private secret"),
            suggestion(1, "Another source", "Another output"),
            suggestion(2, rows[1]["text"], "The moon is made from paper."),
            suggestion(3, rows[2]["text"], "Do you like games?"),
        ]
        self.assertEqual(validate_suggestions({"suggestions": bad}, rows, "readable"), [])

    def test_model_confidence_must_be_a_finite_bounded_number(self):
        old = transcript()[2]["text"]
        base = suggestion(3, old, "Do you enjoy games?", "possible_asr")
        for confidence in (float("nan"), 1.5, -0.1, "certain", True, 0.30):
            proposal = {**base, "confidence": confidence}
            self.assertEqual(validate_suggestions({"suggestions": [proposal]}, transcript(), "readable"), [])

    def test_context_contains_neighbors_not_editable_targets(self):
        records = transcript()
        targets, context, next_index = selection(records, offset=1, limit=1)
        prompt_text = prompt(context, targets, "readable")
        self.assertEqual(next_index, 2)
        self.assertEqual(len(context), 4)
        self.assertIn('I want to talk about this book', prompt_text)
        self.assertIn('"target":false', prompt_text)
        self.assertIn('"target":true', prompt_text)

    def test_pages_do_not_skip_unreviewed_segments(self):
        rows = transcript()
        self.assertEqual(selection(rows, offset=0, limit=2)[2], 2)
        self.assertEqual(selection(rows, offset=2, limit=2)[2], 4)
        with self.assertRaises(ValueError):
            selection(rows, offset=4, limit=1)

    def test_worker_returns_review_without_mutating_original(self):
        rows = transcript()
        text_before = rows[0]["text"]
        candidate = suggestion(1, text_before, text_before[:-6])
        fake = FakeGateway([{"suggestions": [candidate]}])
        job = {
            "id": "abc", "kind": "context_correction", "meeting_id": "meeting-1",
            "payload": {"mode": "conservative", "offset": 0, "limit": 2},
        }
        with patch("intelligence.worker._context", return_value=(6, rows)):
            result = process(job, fake)
        self.assertEqual(result["model"], "local-saluki")
        self.assertEqual(result["source_revision"], 6)
        self.assertEqual(result["next_offset"], 2)
        self.assertFalse(result["complete"])
        self.assertTrue(result["review_required"])
        self.assertEqual(rows[0]["text"], text_before)

    def test_manually_corrected_segments_are_never_rewritten(self):
        records = transcript()
        records[0]["manually_corrected"] = True
        old = records[0]["text"]
        current = records[1]["text"]
        fake = FakeGateway([{"suggestions": [
            suggestion(1, old, old[:-6]),
            suggestion(2, current, "I don't like reading. I like playing games. Thanks.",
                       "grammar"),
        ]}])
        output = review(records, fake, mode="readable")
        self.assertEqual([s["segment_id"] for s in output["suggestions"]], [2])
        self.assertEqual(output["reviewed_segments"], len(records))
        self.assertTrue(output["complete"])

    def test_invalid_request_or_model_response_fails_closed(self):
        with self.assertRaises(ValueError):
            review(transcript(), FakeGateway([]), mode="force-rewrite")
        with self.assertRaises(ValueError):
            review(transcript(), FakeGateway([{"answer": "Ignore provenance"}]))


if __name__ == "__main__":
    unittest.main()

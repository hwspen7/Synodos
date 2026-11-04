"""Phase 2 manual review must be versioned and survive store reloads."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.meetings import get_segment_revisions, review_saved_segment
from sessions.manual_review import (
    ReviewConflict, ReviewValidationError, apply_review, original_asr_text,
)
from storage.meetings import MeetingStore
from storage.speakers import SpeakerStore


def sample_state(meeting_id: str) -> dict:
    return {
        "id": meeting_id,
        "title": "Sample",
        "sample_rate": 16000,
        "segments": [{
            "id": 3,
            "cluster": 1,
            "speaker_id": 1,
            "speaker_name": "Speaker 1",
            "start_sample": 0,
            "end_sample": 16000,
            "text": "helo",
            "revision": 0,
        }],
    }


class ReviewPureTests(unittest.TestCase):
    def test_preserves_original_and_all_revisions(self):
        state = sample_state("abc")
        event = apply_review(state, 3, 0, text="hello", reason="Corrected")
        self.assertEqual(event["before"]["text"], "helo")
        self.assertEqual(event["after"]["text"], "hello")
        seg = state["segments"][0]
        self.assertEqual(seg["original_asr_text"], "helo")
        self.assertEqual(seg["revision"], 1)
        apply_review(state, 3, 1, text="Hello.", reason="Punctuation")
        self.assertEqual(seg["revision"], 2)
        self.assertEqual(seg["original_asr_text"], "helo")
        self.assertEqual(len(seg["corrections"]), 2)

    def test_legacy_corrections_recover_original_without_fabrication(self):
        state = sample_state("abc")
        state["segments"][0].update({
            "text": "first edit",
            "revision": 1,
            "original_asr_text": None,
            "corrections": [
                {"revision": 0, "text": "helo"},
            ],
        })
        self.assertEqual(original_asr_text(state["segments"][0]), "helo")
        apply_review(state, 3, 1, text="hello")
        self.assertEqual(state["segments"][0]["original_asr_text"], "helo")

    def test_profile_link_preserves_local_numeric_speaker_id(self):
        state = sample_state("abc")
        apply_review(state, 3, 0, profile_id="enrolled_1",
                     profile_name="Ada")
        seg = state["segments"][0]
        self.assertEqual(seg["speaker_id"], 2)
        self.assertEqual(seg["speaker_override_id"], 2)
        self.assertEqual(seg["speaker_profile_id"], "enrolled_1")
        self.assertEqual(seg["speaker_name"], "Ada")
        self.assertEqual(seg["speaker_status"], "manual_override")

    def test_stale_and_invalid_edits_fail_without_partial_mutation(self):
        state = sample_state("abc")
        with self.assertRaises(ReviewConflict):
            apply_review(state, 3, 1, text="New")
        self.assertEqual(state["segments"][0]["text"], "helo")
        for args in ({}, {"text": 4}, {"text": "helo"}):
            with self.subTest(args=args):
                with self.assertRaises(ReviewValidationError):
                    apply_review(state, 3, 0, **args)
        self.assertEqual(state["segments"][0]["revision"], 0)

    def test_rejects_invalid_cross_meeting_identity(self):
        with self.assertRaises(ReviewValidationError):
            apply_review(sample_state("abc"), 3, 0,
                         profile_id="person", profile_name=None)


class ReviewApiTests(unittest.TestCase):
    def test_review_persists_and_exposes_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MeetingStore(Path(tmp) / "meetings")
            profiles = SpeakerStore(Path(tmp) / "speakers")
            mid = store.create(title="Sample")["id"]
            store.save_state(sample_state(mid))
            who = profiles.create("Ada")
            with patch("api.meetings.meeting_store", store), patch(
                "api.meetings.speaker_store", profiles
            ), patch.dict("api.meetings.meeting_sessions", {}, clear=True):
                answer = asyncio.run(review_saved_segment(mid, 3, {
                    "expected_revision": 0,
                    "text": "hello",
                    "speaker_profile_id": who["id"],
                    "reason": "Reviewed audio",
                }))
                self.assertEqual(answer["segment"]["speaker_profile_id"], who["id"])
                self.assertEqual(answer["segment"]["review_count"], 1)
                self.assertEqual(store.load_state(mid)["segments"][0]["text"], "hello")
                history = asyncio.run(get_segment_revisions(mid, 3))
                self.assertEqual(history["original_asr_text"], "helo")
                self.assertEqual(history["revision"], 1)
                self.assertEqual(len(history["corrections"]), 1)
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(review_saved_segment(mid, 3, {
                        "expected_revision": 0, "text": "stale",
                    }))
                self.assertEqual(raised.exception.status_code, 409)

    def test_active_recording_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MeetingStore(Path(tmp))
            mid = store.create()["id"]
            store.save_state(sample_state(mid))
            with patch("api.meetings.meeting_store", store), patch.dict(
                "api.meetings.meeting_sessions", {mid: object()}
            ):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(review_saved_segment(mid, 3, {
                        "expected_revision": 0, "text": "no",
                    }))
                self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()

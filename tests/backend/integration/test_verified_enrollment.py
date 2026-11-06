"""Human-reviewed enrollment is explicit and cannot use a mismatched profile."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np

from api.speakers import add_speaker_from_meeting
from storage.meetings import MeetingStore
from sessions.manual_review import apply_review
from storage.speakers import SpeakerStore


class HumanVerifiedEnrollmentTests(unittest.TestCase):
    def test_reviewed_identity_is_required_before_marking_voiceprint_verified(self):
        with tempfile.TemporaryDirectory() as base:
            meetings = MeetingStore(Path(base) / "meetings")
            speakers = SpeakerStore(Path(base) / "speakers")
            who = speakers.create("person")
            another = speakers.create("other")
            mid = meetings.create()["id"]
            state = {
                "id": mid, "title": "Sample",
                "sample_rate": 16000, "segments": [{
                    "id": 1,
                    "cluster": 1,
                    "speaker_id": 1,
                    "speaker_name": "person",
                    "speaker_profile_id": who["id"],
                    "start_sample": 0,
                    "end_sample": 16000 * 6,
                    "emb": np.array([1.0] + [0.0] * 191, dtype=np.float32),
                    "text": "reviewed recording",
                }],
            }
            meetings.save_state(state)
            with patch("api.speakers.meeting_store", meetings), patch(
                "api.speakers.speaker_store", speakers
            ):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(add_speaker_from_meeting(
                        another["id"], {
                            "meeting_id": mid,
                            "segment_ids": [1],
                            "human_verified": True,
                        }
                    ))
                self.assertEqual(captured.exception.status_code, 400)
                with self.assertRaises(HTTPException) as unreviewed:
                    asyncio.run(add_speaker_from_meeting(
                        who["id"], {
                            "meeting_id": mid,
                            "segment_ids": [1],
                            "human_verified": True,
                        },
                    ))
                self.assertEqual(unreviewed.exception.status_code, 400)
                reviewed = meetings.load_state(mid)
                apply_review(reviewed, 1, 0, profile_id=who["id"],
                             profile_name="person", reason="Listened to source audio")
                reviewed.pop("meta", None)
                meetings.save_state(reviewed)
                reply = asyncio.run(add_speaker_from_meeting(
                    who["id"], {
                        "meeting_id": mid,
                        "segment_ids": [1],
                        "human_verified": True,
                    },
                ))
                self.assertEqual(len(reply["vectors"]), 1)
                self.assertTrue(reply["vectors"][0]["human_verified"])
                profile = speakers.get_for_matching([who["id"]])[0]
                self.assertEqual(
                    profile["vectors"][0]["verification_source"],
                    "manual_profile_correction",
                )

    def test_rejects_modified_profile_id_without_a_matching_review_event(self):
        with tempfile.TemporaryDirectory() as base:
            meetings = MeetingStore(Path(base) / "meetings")
            speakers = SpeakerStore(Path(base) / "speakers")
            a = speakers.create("Alice")
            b = speakers.create("Bob")
            mid = meetings.create()["id"]
            state = {
                "id": mid, "title": "Audit", "sample_rate": 16000,
                "segments": [{
                    "id": 2, "cluster": 1, "speaker_id": 1,
                    "start_sample": 0, "end_sample": 16000 * 4,
                    "text": "yes", "revision": 0,
                    "emb": np.ones(192, dtype=np.float32),
                }],
            }
            apply_review(state, 2, 0, profile_id=a["id"],
                         profile_name="Alice")
            state["segments"][0]["speaker_profile_id"] = b["id"]
            meetings.save_state(state)
            with patch("api.speakers.meeting_store", meetings), patch(
                "api.speakers.speaker_store", speakers
            ):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(add_speaker_from_meeting(b["id"], {
                        "meeting_id": mid, "segment_ids": [2],
                        "human_verified": True,
                    }))
            self.assertEqual(captured.exception.status_code, 400)

    def test_rejects_mixed_speaker_audio_even_after_manual_identity_review(self):
        with tempfile.TemporaryDirectory() as base:
            meetings = MeetingStore(Path(base) / "meetings")
            speakers = SpeakerStore(Path(base) / "speakers")
            who = speakers.create("Alice")
            mid = meetings.create()["id"]
            state = {
                "id": mid, "title": "Audit", "sample_rate": 16000,
                "segments": [{
                    "id": 2, "cluster": -3, "speaker_id": -1,
                    "start_sample": 0, "end_sample": 16000 * 4,
                    "speaker_uncertain": True,
                    "speaker_evidence_status": "multi_turn_acoustic_evidence_only",
                    "text": "hi there", "revision": 0,
                    "emb": np.ones(192, dtype=np.float32),
                }],
            }
            apply_review(state, 2, 0, profile_id=who["id"],
                         profile_name="Alice")
            meetings.save_state(state)
            with patch("api.speakers.meeting_store", meetings), patch(
                "api.speakers.speaker_store", speakers
            ):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(add_speaker_from_meeting(who["id"], {
                        "meeting_id": mid, "segment_ids": [2],
                        "human_verified": True,
                    }))
            self.assertEqual(captured.exception.status_code, 400)

    def test_unverified_enrollment_stays_unverified(self):
        with tempfile.TemporaryDirectory() as root:
            store = SpeakerStore(Path(root))
            who = store.create("test")
            output = store.add_vectors(who["id"], [{
                "emb": np.ones(192), "duration_s": 6.,
                "embedding_model": "camplus",
            }])
            self.assertFalse(output[0]["human_verified"])


if __name__ == "__main__":
    unittest.main()

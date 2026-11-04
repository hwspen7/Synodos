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

"""Backend interface and storage regression checks. No inference is performed."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

_ROOT = tempfile.TemporaryDirectory(prefix="meeting-test-")
for _env in ("MEETINGS_DIR", "SPEAKERS_DIR", "HOTWORDS_DIR"):
    os.environ[_env] = os.path.join(_ROOT.name, _env.lower())

from main import app  # noqa: E402
from storage.meetings import MeetingStore  # noqa: E402


class APIContractTests(unittest.TestCase):
    def test_http_route_contract(self) -> None:
        """Check the effective public API, including nested FastAPI routers."""
        expected = {
            ("/api/health", "GET"), ("/api/meetings", "GET"),
            ("/api/meetings", "POST"), ("/api/meetings/{meeting_id}", "GET"),
            ("/api/meetings/{meeting_id}/segments", "GET"),
            ("/api/meetings/{meeting_id}/segments/{segment_id}/review", "PATCH"),
            ("/api/meetings/{meeting_id}/segments/{segment_id}/revisions", "GET"),
            ("/api/meetings/{meeting_id}/segments/{segment_id}/identity-candidate", "GET"),
            ("/api/meetings/{meeting_id}/save", "POST"),
            ("/api/meetings/{meeting_id}", "DELETE"),
            ("/api/meetings/{meeting_id}", "PATCH"),
            ("/api/meetings/{meeting_id}/rename", "POST"),
            ("/api/meetings/{meeting_id}/speakers/label", "POST"),
            ("/api/meetings/{meeting_id}/export", "GET"),
            ("/api/speakers", "GET"), ("/api/speakers", "POST"),
            ("/api/speakers/{speaker_id}", "PATCH"),
            ("/api/speakers/{speaker_id}", "DELETE"),
            ("/api/speakers/{speaker_id}/vectors/record", "POST"),
            ("/api/speakers/{speaker_id}/vectors/meeting", "POST"),
            ("/api/speakers/{speaker_id}/vectors/{vector_id}", "DELETE"),
            ("/api/hotwords", "GET"), ("/api/hotwords", "POST"),
            ("/api/hotwords/{hotword_id}", "PATCH"),
            ("/api/hotwords/{hotword_id}", "DELETE"),
        }
        paths = app.openapi()["paths"]
        actual = {
            (path, method.upper())
            for path, operations in paths.items()
            for method in operations
            if method.upper() in {"GET", "POST", "PATCH", "PUT", "DELETE"}
        }
        self.assertTrue(
            expected <= actual,
            f"Missing operations: {sorted(expected - actual)}",
        )


    def test_speaker_naming_creates_one_library_profile_and_renames_all_turns(self) -> None:
        """A display name survives reload/export without verifying biometrics."""
        from api.meetings import MeetingSpeakerLabelPayload, label_meeting_speaker
        from storage.speakers import SpeakerStore

        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(os.path.join(root, "meetings"))
            library = SpeakerStore(os.path.join(root, "speakers"))
            mid = store.create(title="Planning")["id"]
            state = {
                "id": mid, "title": "Planning", "sample_rate": 16000,
                "merge_gap_s": 2.0,
                "segments": [
                    {
                        "id": 1, "speaker_id": 1, "speaker_name": "Speaker 1",
                        "start_sample": 0, "end_sample": 16000,
                        "cluster": 1, "text": "First remark",
                    },
                    {
                        "id": 2, "speaker_id": 2, "speaker_name": "Speaker 2",
                        "start_sample": 16000, "end_sample": 32000,
                        "cluster": 2, "text": "Second remark",
                    },
                    {
                        "id": 3, "speaker_id": 1, "speaker_name": "Speaker 1",
                        "start_sample": 32000, "end_sample": 48000,
                        "cluster": 1, "text": "More remarks",
                    },
                ],
            }
            store.save_state(state)
            with (patch("api.meetings.meeting_store", store),
                  patch("api.meetings.speaker_store", library),
                  patch("api.meetings.meeting_sessions", {})):
                result = asyncio.run(label_meeting_speaker(
                    mid, MeetingSpeakerLabelPayload(speaker_id=1, name="Alice"),
                ))
                self.assertEqual(result["speaker"]["name"], "Alice")
                self.assertEqual(result["speaker"]["vector_count"], 0)
                first_id = result["speaker"]["id"]
                self.assertEqual(len(library.list_speakers()), 1)
                rows = store.list_segments(mid)
                self.assertEqual([s["speaker_name"] for s in rows],
                                 ["Alice", "Speaker 2", "Alice"])
                self.assertEqual(store.get_meeting(mid)["speaker_labels"]["1"]["name"],
                                 "Alice")
                self.assertIn("Alice", store.export(mid, "md"))
                self.assertIn("Alice", store.export(mid, "json"))
                self.assertIsNone(store.load_state(mid)["segments"][0].get("speaker_profile_id"))

                updated = asyncio.run(label_meeting_speaker(
                    mid, MeetingSpeakerLabelPayload(speaker_id=1, name="Alex"),
                ))
                self.assertEqual(updated["speaker"]["id"], first_id)
                self.assertEqual(library.list_speakers()[0]["name"], "Alex")
                self.assertEqual(len(library.list_speakers()), 1)
                self.assertEqual(store.list_segments(mid)[0]["speaker_name"], "Alex")
                same = asyncio.run(label_meeting_speaker(
                    mid, MeetingSpeakerLabelPayload(speaker_id=1, name="Alex"),
                ))
                self.assertEqual(same["speaker"]["id"], first_id)
                self.assertEqual(len(library.list_speakers()), 1)

    def test_speaker_naming_rejects_active_session(self) -> None:
        from fastapi import HTTPException
        from api.meetings import MeetingSpeakerLabelPayload, label_meeting_speaker
        from storage.speakers import SpeakerStore

        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(os.path.join(root, "meetings"))
            library = SpeakerStore(os.path.join(root, "speakers"))
            mid = store.create()["id"]
            with (patch("api.meetings.meeting_store", store),
                  patch("api.meetings.speaker_store", library),
                  patch("api.meetings.meeting_sessions", {mid: object()})):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(label_meeting_speaker(
                        mid, MeetingSpeakerLabelPayload(speaker_id=1, name="Alice"),
                    ))
                self.assertEqual(captured.exception.status_code, 409)
                self.assertFalse(library.list_speakers())

    def test_health_reports_flat_speech_engine_and_model(self) -> None:
        """The live /api/health handler must not treat settings as a module."""
        from api.health import health
        from speech.config import SpeechSettings

        speech_settings = SpeechSettings(engine="whisper", model="large-v3")
        with patch("api.health.load_speech_settings", return_value=speech_settings):
            payload = asyncio.run(health())

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["realtime"]["asr_engine"], "whisper")
        self.assertEqual(payload["realtime"]["asr_model"], "large-v3")
        self.assertIn("active_sessions", payload)
        self.assertIn("infrastructure", payload)

    def test_websocket_and_openapi(self) -> None:
        """Check the websocket declaration without assuming flattened app.routes."""
        from api.websocket import router as websocket_router

        websocket_paths = {
            route.path for route in websocket_router.routes
            if route.__class__.__name__ == "APIWebSocketRoute"
        }
        self.assertIn("/ws/audio", websocket_paths)
        schema = app.openapi()
        self.assertIn("/api/meetings", schema["paths"])
        self.assertIn("/api/speakers", schema["paths"])

class StorageTests(unittest.TestCase):
    def test_rename_updates_list_resume_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(root)
            mid = store.create(title="Untitled Meeting")["id"]
            state = {
                "id": mid, "title": "Untitled Meeting", "sample_rate": 16000,
                "merge_gap_s": 2.0, "segments": [], "speaker_map": {},
            }
            store.save_state(state)
            renamed = store.rename(mid, "Planning Session")
            self.assertEqual(renamed["title"], "Planning Session")
            self.assertEqual(store.get_meeting(mid)["title"], "Planning Session")
            self.assertEqual(store.list_meetings()[0]["title"], "Planning Session")
            self.assertEqual(store.load_state(mid)["title"], "Planning Session")
            self.assertIn("# Planning Session", store.export(mid, "md"))
            self.assertEqual(store.rename("missing-id", "Any"), None)

    def test_save_resume_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(root)
            mid = store.create(title="Test meeting")["id"]
            state = {
                "id": mid, "title": "Test meeting", "sample_rate": 16000,
                "merge_gap_s": 2.0, "speaker_map": {},
                "segments": [{
                    "id": 1, "start_sample": 0, "end_sample": 16000,
                    "text": "Hello", "cluster": 1, "speaker_id": 1,
                    "speaker_name": "Speaker 1", "emb": [],
                }],
            }
            store.save_state(state)
            self.assertEqual(store.load_state(mid)["segments"][0]["text"], "Hello")
            self.assertIn("Hello", store.export(mid, "md"))
            self.assertIn("Hello", store.export(mid, "json"))


def tearDownModule() -> None:
    _ROOT.cleanup()


if __name__ == "__main__":
    unittest.main()

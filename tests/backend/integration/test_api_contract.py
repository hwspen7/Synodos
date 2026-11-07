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

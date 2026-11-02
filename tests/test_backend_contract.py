"""Backend interface and storage regression checks. No inference is performed."""

from __future__ import annotations

import os
import tempfile
import unittest

_ROOT = tempfile.TemporaryDirectory(prefix="meeting-test-")
for _env in ("MEETINGS_DIR", "SPEAKERS_DIR", "HOTWORDS_DIR"):
    os.environ[_env] = os.path.join(_ROOT.name, _env.lower())

from main import app  # noqa: E402
from storage.meetings import MeetingStore  # noqa: E402


class APIContractTests(unittest.TestCase):
    def test_http_route_contract(self) -> None:
        expected = {
            ("/api/health", "GET"), ("/api/meetings", "GET"),
            ("/api/meetings", "POST"), ("/api/meetings/{meeting_id}", "GET"),
            ("/api/meetings/{meeting_id}/segments", "GET"),
            ("/api/meetings/{meeting_id}/save", "POST"),
            ("/api/meetings/{meeting_id}", "DELETE"),
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
        actual = {
            (r.path, method)
            for r in app.routes
            for method in (getattr(r, "methods", None) or [])
        }
        self.assertTrue(expected <= actual)

    def test_websocket_and_openapi(self) -> None:
        ws_paths = {r.path for r in app.routes if r.__class__.__name__ == "APIWebSocketRoute"}
        self.assertIn("/ws/audio", ws_paths)
        self.assertIn("/api/meetings", app.openapi()["paths"])
        self.assertIn("/api/speakers", app.openapi()["paths"])


class StorageTests(unittest.TestCase):
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

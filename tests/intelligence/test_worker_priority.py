"""An active microphone meeting takes priority over optional Ollama work."""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import unittest
from unittest.mock import patch

from intelligence.worker import realtime_recording_active


class WorkerPriorityTests(unittest.TestCase):
    def test_realtime_health_gate_defers_llm_jobs(self):
        class Response(io.BytesIO):
            status = 200
        payload = json.dumps({"active_meetings": 1}).encode()
        with patch.dict("os.environ", {"MEETING_AI_DEFER_DURING_RECORDING": "1"}), \
             patch("urllib.request.urlopen", return_value=Response(payload)):
            self.assertTrue(realtime_recording_active())

    def test_idle_backend_allows_jobs(self):
        class Response(io.BytesIO):
            status = 200
        with patch.dict("os.environ", {"MEETING_AI_DEFER_DURING_RECORDING": "1"}), \
             patch("urllib.request.urlopen", return_value=Response(b'{"active_meetings":0}')):
            self.assertFalse(realtime_recording_active())

    def test_unreachable_backend_does_not_deadlock_worker(self):
        with patch.dict("os.environ", {"MEETING_AI_DEFER_DURING_RECORDING": "1"}), \
             patch("urllib.request.urlopen", side_effect=OSError("disconnected")):
            self.assertFalse(realtime_recording_active())

    def test_explicit_disable_skips_health_lookup(self):
        with patch.dict("os.environ", {"MEETING_AI_DEFER_DURING_RECORDING": "0"}), \
             patch("urllib.request.urlopen") as fetch:
            self.assertFalse(realtime_recording_active())
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

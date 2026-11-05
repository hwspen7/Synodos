"""No loss of confirmed ASR words during asynchronous speaker splitting."""
from __future__ import annotations

import os
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from sessions.processor import SessionProcessor


class ContentFirstLongTurnTests(unittest.TestCase):
    def test_long_utterance_remains_one_canonical_segment(self):
        original = "Physics was my first love. I still love physics."
        record = {
            "id": 1, "start_sample": 0, "end_sample": 80000,
            "_audio_base_sample": 0, "text": original,
            "words": [], "language": "en", "cluster": -1,
        }
        future = Future()
        future.set_result([
            {"start": 0, "end": 2400, "text": "Physics was my first love."},
            {"start": 2400, "end": 5000, "text": " I still love physics."},
        ])
        proc = object.__new__(SessionProcessor)
        proc.settings = SimpleNamespace(sample_rate=16000)
        proc.segments = [record]
        proc._speaker_tasks = {1: future}
        proc._ecapa_shadow_audio = {}
        proc._acoustic_turn_splits = 0
        proc._next_segment_id = 2
        proc._last_reclustered_segment_id = 1
        proc._enqueue_translation = lambda segment: None
        proc._retry_deferred_speakers = lambda: None

        with patch.dict(os.environ, {"MEETING_CONTENT_FIRST_FUSION": "1"}):
            updated = proc._collect_speakers()

        self.assertTrue(updated)
        self.assertEqual(len(proc.segments), 1)
        self.assertEqual(proc._next_segment_id, 2)
        self.assertEqual(proc.segments[0]["text"], original)
        self.assertEqual(proc.segments[0]["start_sample"], 0)
        self.assertEqual(proc.segments[0]["end_sample"], 80000)
        self.assertEqual(proc.segments[0]["speaker_evidence_status"],
                         "multi_turn_acoustic_evidence_only")
        self.assertEqual(len(proc.segments[0]["speaker_turn_candidate_intervals_ms"]), 2)

    def test_default_mode_is_content_first(self):
        import inspect
        implementation = inspect.getsource(SessionProcessor._collect_speakers)
        self.assertIn('"MEETING_CONTENT_FIRST_FUSION", "1"', implementation)


if __name__ == "__main__":
    unittest.main()

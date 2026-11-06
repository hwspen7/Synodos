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

    def test_consistent_acoustic_subturns_preserve_one_voice_embedding(self):
        original = "I like writing with a clean pen."
        future = Future()
        future.set_result([
            {"start": 0, "end": 1500, "text": "I like writing",
             "emb": [1.0, 0.0], "ambiguous": False},
            {"start": 1500, "end": 3200, "text": " with a clean pen.",
             "emb": [0.99, 0.07], "ambiguous": False},
        ])
        proc = object.__new__(SessionProcessor)
        proc.settings = SimpleNamespace(sample_rate=16000)
        proc.segments = [{
            "id": 1, "start_sample": 0, "end_sample": 51200,
            "_audio_base_sample": 0, "text": original, "cluster": -3,
            "words": [], "language": "en",
        }]
        proc._speaker_tasks = {1: future}
        proc._ecapa_shadow_audio = {}
        proc._next_segment_id = 2
        proc._last_reclustered_segment_id = 1
        proc._enqueue_translation = lambda record: None
        proc._retry_deferred_speakers = lambda: None
        with patch.dict(os.environ, {"MEETING_CONTENT_FIRST_FUSION": "1"}):
            self.assertTrue(proc._collect_speakers())
        self.assertEqual(len(proc.segments), 1)
        self.assertEqual(proc.segments[0]["text"], original)
        self.assertEqual(
            proc.segments[0]["speaker_evidence_status"],
            "multi_turn_same_voice_supported",
        )
        self.assertFalse(proc.segments[0]["speaker_uncertain"])
        self.assertIsNotNone(proc.segments[0]["emb"])

    def test_eight_second_monologue_with_short_unvoiced_fragment_keeps_voice(self):
        # The confirmed ASR turn remains one record even when acoustic turn
        # detection inserts a 350 ms unknown probe between voiced portions.
        original = "前面说过了中间很短然后继续讲下去"
        future = Future()
        future.set_result([
            {"start": 0, "end": 2600, "text": "前面说过了",
             "emb": [1.0, 0.0], "ambiguous": False},
            {"start": 2600, "end": 2950, "text": "中间很短",
             "emb": None, "ambiguous": False, "short_probe": True},
            {"start": 2950, "end": 5600, "text": "然后继续",
             "emb": [.99, .10], "ambiguous": False},
            {"start": 5600, "end": 8000, "text": "讲下去",
             "emb": [.98, -.12], "ambiguous": False},
        ])
        proc = object.__new__(SessionProcessor)
        proc.settings = SimpleNamespace(sample_rate=16000)
        proc.segments = [{
            "id": 1, "start_sample": 0, "end_sample": 128000,
            "_audio_base_sample": 0, "text": original, "cluster": -1,
            "words": [], "language": "zh",
        }]
        proc._speaker_tasks = {1: future}
        proc._ecapa_shadow_audio = {}
        proc._next_segment_id = 2
        proc._last_reclustered_segment_id = 1
        proc._enqueue_translation = lambda record: None
        proc._retry_deferred_speakers = lambda: None
        with patch.dict(os.environ, {"MEETING_CONTENT_FIRST_FUSION": "1"}):
            self.assertTrue(proc._collect_speakers())
        self.assertEqual(len(proc.segments), 1)
        self.assertEqual(proc.segments[0]["text"], original)
        self.assertEqual(proc.segments[0]["start_sample"], 0)
        self.assertEqual(proc.segments[0]["end_sample"], 128000)
        self.assertIsNotNone(proc.segments[0]["emb"])
        self.assertEqual(
            proc.segments[0]["speaker_evidence_status"],
            "multi_turn_same_voice_supported",
        )

    def test_default_mode_is_content_first(self):
        import inspect
        implementation = inspect.getsource(SessionProcessor._collect_speakers)
        self.assertIn('"MEETING_CONTENT_FIRST_FUSION", "1"', implementation)


if __name__ == "__main__":
    unittest.main()

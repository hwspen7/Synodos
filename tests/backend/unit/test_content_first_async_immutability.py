"""Regression for immutable confirmed captions after asynchronous speaker work.

The content-first mode may emit acoustic speaker cues but it must never
rewrite, split, delete or assign a mixed-voice embedding to published ASR text.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.config import SpeechSettings


class ContentFirstAsyncSpeakerTests(unittest.TestCase):
    def _processor(self):
        class Models:
            def embed(self, _audio):
                return None

        with patch(
            "sessions.processor.load_settings",
            return_value=SpeechSettings(engine="whisper", model="large-v3"),
        ):
            processor = SessionProcessor(
                "content-first-immutable", AudioBuffer(16000),
                Settings(), Models(), asyncio.Queue(),
            )
        processor.segments = [{
            "id": 1,
            "_audio_base_sample": 0,
            "start_sample": 0,
            "end_sample": 38400,
            "text": "Hello there. Hi again.",
            "words": [
                {"text": "Hello", "start_ms": 100, "end_ms": 300},
                {"text": " there.", "start_ms": 350, "end_ms": 800},
                {"text": " Hi", "start_ms": 1320, "end_ms": 1600},
                {"text": " again.", "start_ms": 1700, "end_ms": 2260},
            ],
            "language": "en",
            "emb": None, "cluster": -3, "speaker_uncertain": True,
        }]
        processor._next_segment_id = 2
        pieces = [
            {"start": 100, "end": 800, "text": "Hello there.",
             "emb": np.array([1.0, 0.0])},
            {"start": 1320, "end": 2260, "text": " Hi again.",
             "emb": np.array([0.0, 1.0])},
        ]
        done = Future()
        done.set_result(pieces)
        processor._speaker_tasks[1] = done
        return processor

    def test_content_first_keeps_original_confirmed_segment_unchanged(self):
        worker = self._processor()
        before = {
            key: worker.segments[0][key]
            for key in ("id", "text", "start_sample", "end_sample", "words")
        }
        with patch.dict(
            "os.environ", {"MEETING_CONTENT_FIRST_FUSION": "1"}
        ):
            self.assertTrue(worker._collect_speakers())
        self.assertEqual(len(worker.segments), 1)
        self.assertEqual(worker._next_segment_id, 2)
        for key, value in before.items():
            self.assertEqual(worker.segments[0][key], value)
        retained = worker.segments[0]
        self.assertTrue(retained["speaker_uncertain"])
        self.assertIsNone(retained["emb"])
        self.assertEqual(
            retained["speaker_evidence_status"],
            "multi_turn_acoustic_evidence_only",
        )
        self.assertEqual(
            retained["speaker_turn_candidate_intervals_ms"],
            [
                {"start_ms": 100, "end_ms": 800},
                {"start_ms": 1320, "end_ms": 2260},
            ],
        )

    def test_normal_acoustic_split_remains_available_for_comparison(self):
        worker = self._processor()
        with patch.dict(
            "os.environ", {"MEETING_CONTENT_FIRST_FUSION": "0"}
        ):
            self.assertTrue(worker._collect_speakers())
        self.assertEqual(len(worker.segments), 2)
        self.assertEqual(
            [item["text"] for item in worker.segments],
            ["Hello there.", " Hi again."],
        )

    def test_invalid_partition_stays_rejected_in_both_modes(self):
        worker = self._processor()
        future = Future()
        future.set_result([
            {"start": 100, "end": 800,
             "text": "one lost word", "emb": None},
        ])
        worker._speaker_tasks[1] = future
        with patch.dict(
            "os.environ", {"MEETING_CONTENT_FIRST_FUSION": "1"}
        ):
            self.assertTrue(worker._collect_speakers())
        self.assertEqual(len(worker.segments), 1)
        self.assertEqual(worker.segments[0]["text"], "Hello there. Hi again.")
        self.assertEqual(
            worker.segments[0]["speaker_evidence_status"],
            "turn_split_rejected",
        )


if __name__ == "__main__":
    unittest.main()

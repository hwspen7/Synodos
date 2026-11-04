"""Short speaker probes remain opt-in and uncertain by construction."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import os
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.speaker_tasks import submit_speaker_analysis


class ShortProbeTests(unittest.TestCase):
    def test_default_short_turn_does_not_fabricate_voiceprint(self):
        source = {"start": 0, "end": 900, "text": "Yes", "words": []}
        samples = np.ones(14400, dtype=np.float32) * 0.1
        result = submit_speaker_analysis(
            source, samples, lambda _: np.array([1.0, 0.0]), 16000,
        ).result(timeout=5)
        self.assertIsNone(result[0]["emb"])
        self.assertNotIn("short_probe", result[0])

    def test_explicit_short_turn_probe_is_marked_low_support(self):
        source = {"start": 0, "end": 900, "text": "Yes", "words": []}
        samples = np.ones(14400, dtype=np.float32) * 0.1
        result = submit_speaker_analysis(
            source, samples, lambda _: np.array([1.0, 0.0]), 16000,
            minimum_embedding_seconds=0.9,
        ).result(timeout=5)
        self.assertIsNotNone(result[0]["emb"])
        self.assertTrue(result[0]["short_probe"])

    def test_processor_never_treats_short_probe_as_confident(self):
        with patch.dict(os.environ, {"MEETING_SPEAKER_PROBE_MIN_S": "0.9"}):
            processor = SessionProcessor(
                "short-probe", AudioBuffer(16000, max_seconds=30),
                Settings(), object(), asyncio.Queue(),
            )
        self.assertEqual(processor._speaker_min_audio_s, 0.9)
        original = {
            "id": 1, "_audio_base_sample": 0,
            "start_sample": 0, "end_sample": 14400,
            "text": "Yes", "language": "en", "words": [],
            "emb": None, "cluster": -1, "speaker_uncertain": True,
        }
        processor.segments = [original]
        future = Future()
        future.set_result([{
            "start": 0, "end": 900, "text": "Yes",
            "emb": np.array([1.0, 0.0]),
            "ambiguous": False, "short_probe": True,
        }])
        processor._speaker_tasks[1] = future
        self.assertTrue(processor._collect_speakers())
        self.assertIsNotNone(original["emb"])
        self.assertTrue(original["speaker_uncertain"])
        self.assertEqual(original["speaker_evidence_status"], "short_probe")

    def test_invalid_minimum_duration_rejected(self):
        with self.assertRaises(ValueError):
            submit_speaker_analysis(
                {}, np.ones(16000), lambda _: [1, 0], 16000,
                minimum_embedding_seconds=0.5,
            )
        with patch.dict(os.environ, {"MEETING_SPEAKER_PROBE_MIN_S": "0.6"}):
            with self.assertRaisesRegex(ValueError, "0.8..1.2"):
                SessionProcessor(
                    "invalid-short", AudioBuffer(16000, max_seconds=30),
                    Settings(), object(), asyncio.Queue(),
                )


if __name__ == "__main__":
    unittest.main()

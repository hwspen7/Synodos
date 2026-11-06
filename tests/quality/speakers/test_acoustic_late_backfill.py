"""Regression tests for late CAM++ acoustic-timeline reconciliation.

These are deterministic tests of the original live fallback policy. They
never load a speaker model and never guess identities from nearby text.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import unittest

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


def vector(index: int) -> np.ndarray:
    result = np.zeros(192, dtype=np.float32)
    result[index] = 1.0
    return result


def processor() -> SessionProcessor:
    settings = Settings()
    return SessionProcessor(
        "late-acoustics", AudioBuffer(settings.sample_rate, max_seconds=30),
        settings, object(), asyncio.Queue(),
    )


def segment(*, segment_id=1, pending=False, override=False) -> dict:
    return {
        "id": segment_id, "start_sample": 16000,
        "end_sample": 40000, "emb": None, "cluster": -1,
        "speaker_id": -1, "speaker_name": "Unidentified Speaker",
        "speaker_uncertain": pending,
        "speaker_override": override,
        "text": "speech", "language": "en",
    }


class AcousticLateBackfillTests(unittest.TestCase):
    def test_ready_acoustic_windows_backfill_late_missing_segment(self):
        proc = processor()
        ref = vector(3)
        self.assertTrue(proc._acoustic_timeline.add(0, 32000, ref))
        self.assertTrue(proc._acoustic_timeline.add(24000, 56000, ref))
        proc.segments = [segment(pending=True)]
        # No *new* window results in _acoustic_tasks. This previously
        # returned immediately and left the utterance unidentified.
        self.assertEqual(proc._acoustic_tasks, {})
        self.assertTrue(proc._collect_acoustic())
        self.assertTrue(np.allclose(proc.segments[0]["emb"], ref))
        self.assertFalse(proc.segments[0]["speaker_uncertain"])
        self.assertEqual(proc._last_reclustered_segment_id, 0)
        self.assertFalse(proc._collect_acoustic())

    def test_never_backfill_while_turn_analysis_is_in_flight(self):
        proc = processor()
        proc._acoustic_timeline.add(0, 56000, vector(0))
        record = segment(pending=True)
        proc.segments = [record]
        incomplete = Future()
        proc._speaker_tasks[1] = incomplete
        self.assertFalse(proc._collect_acoustic())
        self.assertIsNone(record["emb"])
        proc._speaker_tasks.pop(1)
        self.assertTrue(proc._collect_acoustic())
        self.assertIsNotNone(record["emb"])

    def test_conflicting_acoustic_windows_remain_unidentified(self):
        proc = processor()
        proc._acoustic_timeline.add(0, 32000, vector(0))
        proc._acoustic_timeline.add(24000, 56000, vector(1))
        proc.segments = [segment(pending=True)]
        self.assertFalse(proc._collect_acoustic())
        self.assertIsNone(proc.segments[0]["emb"])
        self.assertTrue(proc.segments[0]["speaker_uncertain"])

    def test_do_not_change_manual_speaker_override(self):
        proc = processor()
        proc._acoustic_timeline.add(0, 56000, vector(2))
        record = segment(override=True)
        record["speaker_id"] = 12
        record["speaker_name"] = "Participant"
        proc.segments = [record]
        self.assertFalse(proc._collect_acoustic())
        self.assertIsNone(record["emb"])
        self.assertEqual(record["speaker_id"], 12)
        self.assertEqual(record["speaker_name"], "Participant")

    def test_do_not_replace_existing_embedding(self):
        proc = processor()
        proc._acoustic_timeline.add(0, 56000, vector(2))
        record = segment()
        record["emb"] = vector(1)
        proc.segments = [record]
        self.assertFalse(proc._collect_acoustic())
        self.assertTrue(np.allclose(record["emb"], vector(1)))


if __name__ == "__main__":
    unittest.main()

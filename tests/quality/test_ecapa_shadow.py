"""Contract tests for optional asynchronous ECAPA shadow operation.

No model download, CUDA, process spawn or stored voiceprint is needed.
"""
from __future__ import annotations

import asyncio
import base64
import os
from unittest.mock import patch
import unittest

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.speaker_shadow import create_shadow_session
from speech.ecapa_shadow_worker import ShadowCentroids, decode_audio


def vector(index: int) -> np.ndarray:
    value = np.zeros(192, dtype=np.float32)
    value[index] = 1
    return value


class FakeShadow:
    def __init__(self):
        self.accepted = []

    def submit(self, label, audio):
        self.accepted.append((label, audio.copy()))
        return True


class ShadowTests(unittest.TestCase):
    def test_opt_in_is_off_by_default(self):
        with patch.dict(os.environ, {"MEETING_ECAPA_SHADOW": "0"}):
            self.assertIsNone(create_shadow_session())

    def test_rank_comparison_is_not_a_calibrated_identity_decision(self):
        state = ShadowCentroids()
        self.assertEqual(state.observe(1, vector(0)), "seeded")
        self.assertEqual(state.observe(1, vector(0)), "no_comparator")
        self.assertEqual(state.observe(2, vector(1)), "seeded")
        self.assertEqual(state.observe(1, vector(0)), "rank_match")
        self.assertEqual(state.observe(1, vector(1)), "rank_conflict")
        self.assertEqual(state.observe(-1, vector(0)), "unlabeled")
        self.assertEqual(state.observe(1, None), "no_embedding")
        self.assertEqual(len(state._centers), 2)

    def test_bounded_queue_drops_instead_of_blocking(self):
        with patch.dict(os.environ, {"MEETING_ECAPA_SHADOW": "1"}):
            shadow = create_shadow_session()
        self.assertIsNotNone(shadow)
        # Suppress the actual worker for a deterministic queue test.
        shadow._thread = object()
        audio = np.ones(16000 * 2, dtype=np.float32) * 0.1
        try:
            for _ in range(6):
                self.assertTrue(shadow.submit(1, audio))
            self.assertFalse(shadow.submit(1, audio))
            result = shadow.status()
            self.assertEqual(result["submitted"], 6)
            self.assertEqual(result["skipped_queue_full"], 1)
            self.assertEqual(result["remaining"], 6)
            self.assertNotIn("embedding", result)
        finally:
            shadow.close()
        self.assertFalse(shadow.submit(1, audio))

    def test_pcm16_protocol_validates_sample_rate_and_shape(self):
        pcm = np.full(32000, 8000, dtype="<i2").tobytes()
        data = {
            "sample_rate": 16000,
            "pcm16_b64": base64.b64encode(pcm).decode("ascii"),
        }
        result = decode_audio(data)
        self.assertEqual(result.shape, (32000,))
        self.assertAlmostEqual(float(result[0]), 8000 / 32768, places=4)
        with self.assertRaises(ValueError):
            decode_audio({**data, "sample_rate": 48000})
        with self.assertRaises(ValueError):
            decode_audio({**data, "pcm16_b64": "!invalid!"})

    def test_processor_only_observes_and_never_changes_published_labels(self):
        settings = Settings()
        processor = SessionProcessor(
            "shadow-safe", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue()
        )
        spy = FakeShadow()
        processor._ecapa_shadow = spy
        audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        processor._ecapa_shadow_audio = {1: audio.copy()}
        record = {
            "id": 1, "start_sample": 0, "end_sample": len(audio),
            "text": "hello", "language": "en", "emb": vector(0),
            "cluster": 0, "speaker_id": 7, "speaker_name": "Speaker 7",
            "speaker_uncertain": False,
        }
        processor.segments = [record]
        processor._rebuild_and_push()
        self.assertEqual(len(spy.accepted), 1)
        self.assertEqual(spy.accepted[0][0], 7)
        self.assertEqual(record["speaker_id"], 7)
        self.assertEqual(record["speaker_name"], "Speaker 7")
        self.assertEqual(record["text"], "hello")
        self.assertNotIn("ecapa_emb", record)
        processor._rebuild_and_push()
        self.assertEqual(len(spy.accepted), 1)

    def test_uncertain_and_manual_override_are_never_shadow_enrolled(self):
        settings = Settings()
        processor = SessionProcessor(
            "shadow-unknown", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue()
        )
        spy = FakeShadow()
        processor._ecapa_shadow = spy
        audio = np.ones(16000 * 3, dtype=np.float32) * .1
        processor._ecapa_shadow_audio = {1: audio, 2: audio}
        processor.segments = [
            {"id": 1, "speaker_id": 1, "speaker_name": "Speaker 1",
             "speaker_uncertain": True, "emb": vector(0), "cluster": 0,
             "text": "unverified", "start_sample": 0, "end_sample": 48000},
            {"id": 2, "speaker_id": 2, "speaker_name": "Manual",
             "speaker_override": True, "speaker_override_id": 2,
             "speaker_override_name": "Manual", "emb": vector(1), "cluster": 1,
             "text": "manual", "start_sample": 48000, "end_sample": 96000},
        ]
        processor._rebuild_and_push()
        self.assertEqual(spy.accepted, [])


if __name__ == "__main__":
    unittest.main()

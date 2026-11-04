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
from sessions.speaker_shadow import create_shadow_session, select_shadow_window
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

    def test_quiet_speech_later_in_turn_is_selected_not_first_six_seconds(self):
        audio = np.zeros(16000 * 12, dtype=np.float32)
        audio[16000 * 7:16000 * 11] = 0.003
        excerpt = select_shadow_window(audio)
        self.assertIsNotNone(excerpt)
        self.assertEqual(len(excerpt), 16000 * 3)
        self.assertGreater(float(np.sqrt(np.mean(excerpt * excerpt))), 0.001)
        self.assertLess(float(np.sqrt(np.mean(excerpt * excerpt))), 0.008)

        with patch.dict(os.environ, {"MEETING_ECAPA_SHADOW": "1"}):
            shadow = create_shadow_session()
        self.assertIsNotNone(shadow)
        shadow._thread = object()  # No real model process in contract tests.
        try:
            self.assertTrue(shadow.submit(2, excerpt))
            report = shadow.status()
            self.assertEqual(report["accepted_quiet_speech"], 1)
            self.assertEqual(report["observed_camplus_speaker_count"], 1)
            self.assertEqual(report["skipped_low_energy"], 0)
        finally:
            shadow.close()

    def test_true_silence_and_impulses_are_not_selected(self):
        self.assertIsNone(
            select_shadow_window(np.zeros(16000 * 8, dtype=np.float32))
        )
        transient = np.zeros(16000 * 6, dtype=np.float32)
        transient[:1600] = 0.04
        self.assertIsNone(select_shadow_window(transient))
        self.assertIsNone(select_shadow_window(
            np.ones(16000 * 2, dtype=np.float32) * float("nan")
        ))

    def test_processor_uses_active_excerpt_not_silent_opening(self):
        settings = Settings()
        processor = SessionProcessor(
            "shadow-active", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        spy = FakeShadow()
        processor._ecapa_shadow = spy
        raw = np.zeros(16000 * 12, dtype=np.float32)
        raw[16000 * 7:16000 * 11] = .003
        processor._ecapa_shadow_audio = {1: raw}
        processor.segments = [{
            "id": 1, "start_sample": 0, "end_sample": len(raw),
            "text": "audio segment", "language": "en",
            "emb": vector(0), "cluster": 0, "speaker_id": 1,
            "speaker_name": "Speaker 1", "speaker_uncertain": False,
        }]
        processor._rebuild_and_push()
        self.assertEqual(len(spy.accepted), 1)
        self.assertEqual(len(spy.accepted[0][1]), 48000)
        self.assertGreater(float(np.sqrt(np.mean(
            spy.accepted[0][1] ** 2
        ))), 0.001)
        self.assertEqual(processor.segments[0]["speaker_id"], 1)

    def test_long_meeting_samples_continue_after_first_64_segments(self):
        settings = Settings()
        processor = SessionProcessor(
            "long-shadow", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        spy = FakeShadow()
        processor._ecapa_shadow = spy
        audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        for index in range(70):
            sid = index + 1
            start = index * 11 * 16000
            processor.segments = [{
                "id": sid, "start_sample": start,
                "end_sample": start + len(audio),
                "speaker_id": 1, "speaker_name": "Speaker 1",
                "speaker_uncertain": False, "emb": vector(0),
                "cluster": 0, "text": "voice", "language": "en",
            }]
            processor._ecapa_shadow_audio = {sid: audio.copy()}
            processor._submit_shadow_samples()
        self.assertEqual(len(spy.accepted), 70)
        self.assertEqual(processor._ecapa_shadow_sample_count, 70)
        self.assertTrue(all(row[0] == 1 for row in spy.accepted))

    def test_sampling_cadence_preserves_new_speaker_prototype(self):
        settings = Settings()
        processor = SessionProcessor(
            "cadence-shadow", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        spy = FakeShadow()
        processor._ecapa_shadow = spy
        audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        for sid, label, start in [(1, 1, 0), (2, 1, 16000), (3, 2, 32000)]:
            processor.segments = [{
                "id": sid, "start_sample": start,
                "end_sample": start + len(audio),
                "speaker_id": label, "speaker_name": f"Speaker {label}",
                "speaker_uncertain": False, "emb": vector(label),
                "cluster": label, "text": "voice", "language": "en",
            }]
            processor._ecapa_shadow_audio = {sid: audio.copy()}
            processor._submit_shadow_samples()
        self.assertEqual([item[0] for item in spy.accepted], [1, 2])

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

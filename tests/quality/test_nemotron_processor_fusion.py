"""Integration contracts for published Nemotron/CAM++ speaker projections."""
from __future__ import annotations

import asyncio
import unittest

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


class FakeAcoustics:
    def __init__(self, activity):
        self.values = activity

    def activity_snapshot(self):
        return self.values


def processor_for_test():
    settings = Settings(
        asr_model="mock-asr",
        fsmn_vad_model="mock-vad",
        punc_model="mock-punc",
        spk_model="mock-spk",
    )
    processor = SessionProcessor(
        "fusion-case", AudioBuffer(16000, max_seconds=30),
        settings, object(), asyncio.Queue(),
    )
    processor.segments = [{
        "id": 1,
        "_audio_base_sample": 0,
        "start_sample": 0,
        "end_sample": 16000,
        "text": "Some valid speech",
        "words": [],
        "language": "en",
        "emb": np.array([1., 0.], dtype=np.float32),
        "cluster": 0,
        "speaker_id": 17,
        "speaker_name": "Speaker 17",
    }]
    return processor


class LiveFusionTests(unittest.TestCase):
    def test_independent_activity_corrects_caption_but_not_acoustic_cluster(self):
        processor = processor_for_test()
        activity = np.zeros((100, 8), dtype=np.float32)
        activity[:, 2] = 0.95
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], 3)
        self.assertEqual(processor.segments[0]["speaker_id"], 17)
        self.assertEqual(processor.segments[0]["cluster"], 0)
        state = processor.snapshot_state()
        self.assertEqual(state["segments"][0]["speaker_id"], 3)
        self.assertEqual(
            state["segments"][0]["speaker_diarization_source"], "nemotron3",
        )
        self.assertGreater(state["segments"][0]["speaker_activity_score"], .9)

    def test_manual_override_cannot_be_replaced_by_activity(self):
        processor = processor_for_test()
        processor.segments[0].update({
            "speaker_override": True,
            "speaker_override_id": 8,
            "speaker_override_name": "Manual",
            "saved_nemotron_id": 3,
        })
        activity = np.zeros((100, 8), dtype=np.float32)
        activity[:, 0] = 0.98
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertNotIn(1, processor._independent_published)
        state = processor.snapshot_state()
        self.assertEqual(state["segments"][0]["speaker_id"], 8)

    def test_unscored_activity_does_not_fabricate_speaker(self):
        processor = processor_for_test()
        processor.segments[0]["speaker_id"] = -1
        processor.segments[0]["speaker_name"] = "Unidentified Speaker"
        processor.independent_diarization = FakeAcoustics(
            np.zeros((25, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published, {})


if __name__ == "__main__":
    unittest.main()

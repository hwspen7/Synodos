"""Deterministic tests for source-independent Nemotron streaming and fusion.

No model weights, GPU, network or human reference audio are required.
"""
from __future__ import annotations

import threading
import unittest

import numpy as np

from speech.diarization.nemotron_activity import (
    activity_turns, choose_caption_speaker, validate_activity,
)
from speech.diarization.nemotron_stream import NemotronRealtimeWorker, StreamingChunks


class FakeProcessor:
    num_samples_first_audio_chunk = 16000
    num_samples_per_audio_chunk = 16000
    num_mel_frames_per_step = 1

    @staticmethod
    def audio_chunk_start(frame):
        return 16000 * frame


def fake_runner():
    processor = FakeProcessor()
    seen = []

    def infer(audio, first, last, cache):
        index = 1 if cache is None else cache + 1
        seen.append((len(audio), first, last, cache))
        count = len(audio) // 160
        logits = np.full((count, 8), -9.0, dtype=np.float32)
        logits[:, index % 2] = 9.0
        return logits, index

    stream = StreamingChunks(processor, infer)

    def extract(raw, duration_ms):
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        return activity_turns(scores)

    return stream, extract, seen


class NemotronActivityTests(unittest.TestCase):
    def test_primary_activity_uses_all_frames_and_is_not_cosine(self):
        data = np.zeros((150, 8), dtype=np.float32)
        data[:, 2] = 0.94
        data[:, 4] = 0.2
        outcome = choose_caption_speaker(data, 100, 1100)
        self.assertEqual(outcome.speaker_id, 3)
        self.assertEqual(outcome.status, "acoustic_primary")
        self.assertEqual(outcome.as_dict()["score_kind"],
                         "model_activity_not_identity_probability")

    def test_insufficient_or_ambiguous_activity_does_not_force_identity(self):
        data = np.zeros((100, 8), dtype=np.float32)
        data[:, 0], data[:, 1] = 0.8, 0.78
        self.assertIsNone(choose_caption_speaker(data, 0, 1000).speaker_id)
        self.assertEqual(
            choose_caption_speaker(data, 0, 5000).status,
            "insufficient_timeline",
        )

    def test_overlap_preserves_both_active_channels(self):
        data = np.zeros((10, 8), dtype=np.float32)
        data[:, 0] = 0.95
        data[3:8, 1] = 0.88
        turns = activity_turns(data)
        self.assertEqual(turns, [
            {"start_ms": 0, "end_ms": 100, "speaker_id": 1},
            {"start_ms": 30, "end_ms": 80, "speaker_id": 2},
        ])
        with self.assertRaises(ValueError):
            validate_activity(np.full((1, 8), -1.0))

    def test_first_intermediate_and_final_chunks_share_cache(self):
        stream, extract, seen = fake_runner()
        stream.accept(np.zeros(25000, dtype=np.float32))
        stream.accept(np.zeros(15000, dtype=np.float32))
        stream.finish()
        self.assertEqual([item[0] for item in seen],
                         [16000, 16000, 8000])
        self.assertEqual([item[3] for item in seen], [None, 1, 2])
        self.assertTrue(seen[0][1])
        self.assertTrue(seen[-1][2])
        self.assertEqual(stream.received_samples, 40000)
        self.assertEqual(stream.scored_frames, 250)
        self.assertTrue(extract(stream.raw_logits(), 2500))

    def test_thread_receives_audio_independently_of_asr(self):
        observed = []
        stream, extractor, seen = fake_runner()
        worker = NemotronRealtimeWorker(
            runner_factory=lambda: (stream, extractor),
            on_update=lambda: observed.append(1),
            queue_frames=10,
        )
        worker.start()
        for _ in range(3):
            self.assertTrue(worker.feed(np.zeros(8000, dtype=np.float32)))
        self.assertTrue(worker.finish(timeout=5))
        report = worker.report()
        self.assertTrue(report["complete"])
        self.assertEqual(report["processed_audio_samples"], 24000)
        self.assertTrue(report["cross_chunk_identity_cache"])
        self.assertGreater(report["scored_frames"], 0)
        self.assertTrue(observed)

    def test_micro_frames_are_batched_without_dropping_the_stop_tail(self):
        stream, extractor, _ = fake_runner()
        worker = NemotronRealtimeWorker(
            runner_factory=lambda: (stream, extractor), queue_frames=10,
        )
        # 128 microphone packets are now only five 200ms queue entries,
        # not 128 waiting items. Preserve all 16kHz samples.
        for _ in range(125):
            self.assertTrue(worker.feed(np.full(128, 0.2, dtype=np.float32)))
        self.assertEqual(worker.report()["max_queued_pcm_frames"], 5)
        self.assertEqual(worker.report()["pending_pcm_samples"], 0)
        worker.start()
        self.assertTrue(worker.finish(timeout=5))
        self.assertEqual(worker.report()["processed_audio_samples"], 16000)
        self.assertEqual(worker.report()["input_audio_samples"], 16000)
        self.assertFalse(worker.feed(np.zeros(1600, dtype=np.float32)))

        # A nonintegral number of blocks must flush its exact remaining PCM.
        tail_stream, tail_extractor, _ = fake_runner()
        tail_worker = NemotronRealtimeWorker(
            runner_factory=lambda: (tail_stream, tail_extractor),
            queue_frames=10,
        )
        for _ in range(27):
            self.assertTrue(tail_worker.feed(np.full(128, 0.1, dtype=np.float32)))
        self.assertEqual(tail_worker.report()["pending_pcm_samples"], 256)
        tail_worker.start()
        self.assertTrue(tail_worker.finish(timeout=5))
        self.assertEqual(tail_worker.report()["processed_audio_samples"], 3456)
        self.assertEqual(tail_worker.report()["input_audio_samples"], 3456)

    def test_queue_exhaustion_fails_without_claiming_complete_audio(self):
        stream, extractor, _ = fake_runner()
        worker = NemotronRealtimeWorker(
            runner_factory=lambda: (stream, extractor), queue_frames=1,
        )
        self.assertTrue(worker.feed(np.zeros(3200, dtype=np.float32)))
        self.assertFalse(worker.feed(np.zeros(3200, dtype=np.float32)))
        self.assertEqual(worker.report()["state"], "failed")
        self.assertFalse(worker.report()["complete"])

    def test_no_partial_result_is_labeled_complete_on_worker_failure(self):
        def broken():
            raise RuntimeError("Injected failure")

        worker = NemotronRealtimeWorker(runner_factory=broken)
        worker.start()
        worker.feed(np.zeros(1600, dtype=np.float32))
        self.assertFalse(worker.finish(timeout=5))
        report = worker.report()
        self.assertFalse(report["complete"])
        self.assertEqual(report["state"], "failed")
        self.assertFalse(report["turns"])


if __name__ == "__main__":
    unittest.main()

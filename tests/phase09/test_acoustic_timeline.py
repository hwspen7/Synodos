"""An acoustic timeline is independent of Whisper sentence boundaries."""
import asyncio
from concurrent.futures import Future
import unittest
from unittest.mock import patch

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.acoustic_timeline import AcousticTimeline


class AcousticTimelineTests(unittest.TestCase):
    def test_short_transcript_uses_continuous_audio_evidence(self):
        timeline = AcousticTimeline(16000)
        timeline.add(0, 32000, [1, 0])
        result = timeline.embedding_for(12000, 18000)
        self.assertIsNotNone(result)
        self.assertGreater(result[0], 0.99)

    def test_different_speakers_are_not_averaged_into_one(self):
        timeline = AcousticTimeline(16000)
        timeline.add(0, 24000, [1, 0])
        timeline.add(16000, 40000, [0, 1])
        self.assertIsNone(timeline.embedding_for(15000, 24000))

    def test_silent_or_invalid_embedding_is_rejected(self):
        timeline = AcousticTimeline(16000)
        self.assertFalse(timeline.add(0, 32000, None))
        self.assertFalse(timeline.add(0, 32000, [float("nan")]))
        self.assertIsNone(timeline.embedding_for(0, 16000))

    def test_uncovered_segment_is_unidentified(self):
        timeline = AcousticTimeline(16000)
        timeline.add(0, 32000, [1, 0])
        self.assertIsNone(timeline.embedding_for(48000, 64000))

    def test_processor_schedules_windows_before_whisper_sentence_is_known(self):
        settings = Settings()
        buffer = AudioBuffer(16000, max_seconds=30)
        buffer.append(np.ones(16000 * 6, dtype=np.float32) * 0.05)
        class SpeakerModel:
            def embed(self, audio):
                return np.asarray([1., 0.], dtype=np.float32)

        with patch.dict("os.environ", {"MEETING_ASR_ENGINE": "whisper"}):
            processor = SessionProcessor(
                "independent-windows", buffer, settings, SpeakerModel(), asyncio.Queue(),
            )
        def finished(start, data, _embed, _sr):
            future = Future()
            future.set_result((start, len(data), np.asarray([1., 0.], dtype=np.float32)))
            return future
        with patch("sessions.processor.submit_acoustic_window", side_effect=finished):
            processor._schedule_acoustic(0, np.ones(16000 * 6, dtype=np.float32))
        self.assertGreaterEqual(len(processor._acoustic_tasks), 2)
        self.assertFalse(processor.segments)
        processor._collect_acoustic()
        self.assertTrue(processor._acoustic_timeline.windows)

    def test_processor_backfills_short_segment_without_inventing_text(self):
        settings = Settings()
        buffer = AudioBuffer(16000, max_seconds=30)
        with patch.dict("os.environ", {"MEETING_ASR_ENGINE": "whisper"}):
            processor = SessionProcessor(
                "short-utterance", buffer, settings, object(), asyncio.Queue(),
            )
        processor.segments.append({
            "id": 1, "start_sample": 16000, "end_sample": 20000,
            "emb": None, "cluster": -11, "text": "Yes.",
            "speaker_id": -1, "speaker_name": "Unidentified Speaker",
        })
        future = Future()
        future.set_result((0, 32000, np.asarray([1., 0.], dtype=np.float32)))
        processor._acoustic_tasks[0] = future
        self.assertTrue(processor._collect_acoustic())
        self.assertIsNotNone(processor.segments[0]["emb"])
        self.assertEqual(processor.segments[0]["text"], "Yes.")


if __name__ == "__main__":
    unittest.main()

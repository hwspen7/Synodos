"""Speech text must survive unavailable speaker embeddings or clustering."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import Mock

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


class ModelsWithMissingEmbeddings:
    def transcribe_sentences(self, audio, hotword=None):
        return [{"start": 0, "end": 900, "text": "Confirmed words"}]

    def embed(self, audio):
        return None

    def cluster(self, embeddings):
        raise AssertionError("No embeddings should reach the clustering model")


class ModelsWithClusteringFailure(ModelsWithMissingEmbeddings):
    def embed(self, audio):
        return np.array([0.3, 0.4], dtype=np.float32)

    def cluster(self, embeddings):
        raise RuntimeError("Injected clustering failure")


class TranscriptionFallbackTests(unittest.TestCase):
    @staticmethod
    def make_processor(models):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate)
        buffer.append(np.ones(settings.sample_rate, dtype=np.float32) * 0.1)
        return SessionProcessor(
            session_id="test-segment-fallback",
            buffer=buffer,
            settings=settings,
            models=models,
            outbox=asyncio.Queue(),
        )

    def test_missing_embedding_keeps_recognized_words_and_can_resume(self):
        processor = self.make_processor(ModelsWithMissingEmbeddings())
        self.assertTrue(processor._cycle(final=True))
        self.assertEqual(processor.segments[0]["text"], "Confirmed words")
        self.assertIsNone(processor.segments[0]["emb"])
        self.assertEqual(processor.segments[0]["speaker_name"], "Unidentified Speaker")
        saved = processor.snapshot_state()
        restored = self.make_processor(ModelsWithMissingEmbeddings())
        restored.load_state(saved)
        self.assertIsNone(restored.segments[0]["emb"])
        self.assertEqual(restored.segments[0]["text"], "Confirmed words")

    def test_clustering_failure_still_publishes_confirmed_words(self):
        processor = self.make_processor(ModelsWithClusteringFailure())
        self.assertTrue(processor._cycle(final=True))
        self.assertEqual(processor.segments[0]["text"], "Confirmed words")
        self.assertEqual(processor.segments[0]["speaker_name"], "Unidentified Speaker")

    def test_save_write_failure_reaches_waiting_ticket(self):
        processor = self.make_processor(ModelsWithMissingEmbeddings())
        processor.meeting_id = "a" * 32
        processor.meeting_store = Mock()
        processor.meeting_store.save_state.side_effect = OSError("disk full")
        ticket = processor._save_queue.submit()
        self.assertTrue(processor._drain_save_requests())
        self.assertTrue(ticket.completed.wait(0.01))
        self.assertFalse(ticket.succeeded)

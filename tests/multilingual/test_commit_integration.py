"""Regression tests for ASR confirmation and audio retention."""
import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.contracts import Recognition, Utterance


class FakeMultilingualModel:
    def __init__(self, utterances):
        self.utterances = utterances

    def recognize_multilingual(self, audio, **kwargs):
        return Recognition(tuple(self.utterances), "en", engine="whisper", speech_detected=True)

    def embed(self, audio):
        return None

    def cluster(self, embeddings):
        raise AssertionError("No embeddings should reach cluster backend")


class CommitIntegrationTests(unittest.TestCase):
    def processor(self, utterances):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate, max_seconds=60)
        buffer.append(np.ones(12 * settings.sample_rate, dtype=np.float32) * .1)
        with patch.dict("os.environ", {
            "MEETING_ASR_ENGINE": "whisper", "MEETING_ASR_LANGUAGE": "en",
            "MEETING_TRANSLATE_TO": "", "MEETING_NATIVE_STREAMING": "false",
        }):
            return SessionProcessor(
                session_id="dedup-test",
                buffer=buffer,
                settings=settings,
                models=FakeMultilingualModel(utterances),
                outbox=asyncio.Queue(),
            )

    def test_reprocessed_absolute_span_is_not_confirmed_twice(self):
        p = self.processor([Utterance("Welcome.", 0, 1000, "en")])
        self.assertTrue(p._cycle(final=False))
        self.assertEqual(len(p.segments), 1)
        p.buffer.append(np.ones(12 * p.settings.sample_rate, dtype=np.float32) * .1)
        p._cycle(final=False)
        self.assertEqual(len(p.segments), 2)
        self.assertLessEqual(p.segments[0]["end_sample"], p.segments[1]["start_sample"])

    def test_late_confirmed_phrase_does_not_drop_earlier_unconfirmed_audio(self):
        utterances = [
            Utterance("Earlier speech", 0, 11000, "en"),
            Utterance("Later recognized phrase", 2000, 4000, "en"),
        ]
        p = self.processor(utterances)
        p._cycle(final=False)
        self.assertEqual(p.buffer.start_offset(), 0)
        self.assertEqual(len(p.segments), 1)
        self.assertEqual(p.segments[0]["text"], "Later recognized phrase")

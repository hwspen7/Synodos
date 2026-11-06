"""Ensure multilingual inference uses ModelManager's public scheduling gateway."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from inference.models import ModelManager
from speech.config import SpeechSettings
from speech.contracts import Recognition, Utterance


class InlineScheduler:
    def __init__(self):
        self.submissions = []

    def submit(self, name, callback, priority=0):
        self.submissions.append((name, priority))
        return callback()


class SchedulerContractTests(unittest.TestCase):
    def test_gateway_schedules_multilingual_asr_without_exposing_private_fields(self):
        manager = object.__new__(ModelManager)
        manager._infer_lock = threading.Lock()
        manager._scheduler = InlineScheduler()
        expected = Recognition((Utterance("hello", 0, 900, "en"),), "en", engine="whisper")
        audio = np.zeros(16000, dtype=np.float32)
        config = SpeechSettings(engine="whisper")
        with patch("speech.engine.recognize", return_value=expected) as recognize:
            result = manager.recognize_multilingual(
                audio, language="en", terms="technical term", config=config
            )
        self.assertIs(result, expected)
        self.assertEqual(manager._scheduler.submissions, [("multilingual_asr", 0)])
        recognize.assert_called_once_with(
            audio, language="en", terms="technical term", config=config
        )

    def test_session_processor_calls_public_gateway_for_multilingual_asr(self):
        import asyncio
        from audio.buffer import AudioBuffer
        from core.config import Settings
        from sessions.processor import SessionProcessor

        class MultilingualModelStub:
            def recognize_multilingual(self, samples, *, language, terms, config):
                return Recognition(
                    (Utterance("Multilingual words", 0, 900, "en"),),
                    "en", engine="whisper"
                )

            def embed(self, audio):
                return None

        with patch.dict("os.environ", {
            "MEETING_ASR_ENGINE": "whisper",
            "MEETING_NATIVE_STREAMING": "false",
        }):
            settings = Settings()
            buffer = AudioBuffer(settings.sample_rate)
            # This contract verifies the gateway, not acceptance of an
            # invented transcript on digital silence. Use non-silent PCM so
            # the ASR hallucination veto does not correctly reject the stub.
            buffer.append(np.full(settings.sample_rate, .03, dtype=np.float32))
            processor = SessionProcessor(
                session_id="public-gateway-test",
                buffer=buffer,
                settings=settings,
                models=MultilingualModelStub(),
                outbox=asyncio.Queue(),
            )
            self.assertTrue(processor._cycle(final=True))
            self.assertEqual(processor.segments[0]["text"], "Multilingual words")
            self.assertEqual(processor.segments[0]["language"], "en")

    def test_legacy_fallback_tests_are_independent_of_selected_engine(self):
        from tests.backend.unit.test_transcription_fallback import (
            ModelsWithMissingEmbeddings, TranscriptionFallbackTests
        )
        from unittest.mock import patch as mock_patch
        with mock_patch.dict("os.environ", {"MEETING_ASR_ENGINE": "whisper"}):
            processor = TranscriptionFallbackTests.make_processor(ModelsWithMissingEmbeddings())
            self.assertEqual(processor._speech_settings.engine, "legacy")
            self.assertTrue(processor._cycle(final=True))
            self.assertEqual(processor.segments[0]["text"], "Confirmed words")


if __name__ == "__main__":
    unittest.main()

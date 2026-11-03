"""Phase 07E autosave integration using the existing MeetingStore contract."""
import asyncio
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


class SimpleModel:
    def transcribe_sentences(self, audio, hotword=None):
        return [{"start": 0, "end": 900, "text": "Confirmed words"}]

    def embed(self, audio):
        return None

    def cluster(self, embeddings):
        raise AssertionError("No embeddings should be clustered")


class AutosaveIntegrationTests(unittest.TestCase):
    def make_processor(self, interval):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate, max_seconds=30)
        buffer.append(np.full(settings.sample_rate * 12, .1, dtype=np.float32))
        store = Mock()
        with patch.dict("os.environ", {
            "MEETING_ASR_ENGINE": "legacy", "MEETING_NATIVE_STREAMING": "false",
            "MEETING_PHASE07_AUTOSAVE_SECONDS": str(interval),
        }):
            processor = SessionProcessor("phase07-test", buffer, settings, SimpleModel(),
                                         asyncio.Queue(), meeting_store=store,
                                         meeting_id="a" * 32)
        return processor, store

    def test_confirmed_text_is_autosaved(self):
        processor, store = self.make_processor(15)
        processor._phase07_last_autosave -= 20
        self.assertTrue(processor._cycle(final=False))
        store.save_state.assert_called_once()
        saved = store.save_state.call_args.args[0]
        self.assertEqual(saved["segments"][0]["text"], "Confirmed words")
        self.assertEqual(processor._phase07_saved_segment_id, 1)

    def test_disable_periodic_saves_without_disabling_manual_save(self):
        processor, store = self.make_processor(0)
        processor._phase07_last_autosave -= 20
        self.assertTrue(processor._cycle(final=False))
        store.save_state.assert_not_called()
        self.assertTrue(processor._do_save())
        store.save_state.assert_called_once()

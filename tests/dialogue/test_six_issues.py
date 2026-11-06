"""End-of-stream, independent speaker updates, and translation isolation."""
import asyncio
from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.contracts import Recognition, Utterance, Word


class LegacyTailModel:
    def __init__(self):
        self.calls = 0

    def transcribe_sentences(self, audio, hotword=None):
        self.calls += 1
        return [{"start": 0, "end": min(90, round(len(audio) / 16)),
                 "text": "Short trailing words"}]

    def embed(self, audio):
        return None


class MultilingualModel:
    def __init__(self, utterances):
        self.utterances = utterances

    def recognize_multilingual(self, audio, **kwargs):
        return Recognition(tuple(self.utterances), language="en", engine="whisper", speech_detected=True)

    def embed(self, audio):
        return None

    def cluster(self, embeddings):
        return np.arange(len(embeddings), dtype=int)


def make_processor(model, waveform, engine):
    settings = Settings()
    buffer = AudioBuffer(settings.sample_rate, max_seconds=30)
    buffer.append(np.asarray(waveform, dtype=np.float32))
    with patch.dict("os.environ", {
        "MEETING_ASR_ENGINE": engine,
        "MEETING_TRANSLATE_TO": "",
        "MEETING_NATIVE_STREAMING": "false",
    }):
        return SessionProcessor("phase09-test", buffer, settings, model, asyncio.Queue())


class Phase09PipelineTests(unittest.TestCase):
    def test_subsecond_tail_is_always_submitted_to_final_asr(self):
        model = LegacyTailModel()
        processor = make_processor(model, np.ones(1600, dtype=np.float32), "legacy")
        processor._finalize()
        self.assertEqual(model.calls, 1)
        self.assertEqual(processor.segments[0]["text"], "Short trailing words")

    def test_same_audio_endpoint_is_not_reinferred(self):
        model = LegacyTailModel()
        processor = make_processor(model, np.ones(16000 * 12, dtype=np.float32), "legacy")
        processor._cycle(final=False)
        before = model.calls
        processor._cycle(final=False)
        self.assertEqual(model.calls, before)

    def test_async_acoustic_turns_split_word_aligned_speaker_text(self):
        words = (
            Word("Alpha", 0, 1800), Word("beta", 2000, 3800),
            Word("gamma", 4000, 5800), Word("delta", 6000, 7800),
        )
        model = MultilingualModel([Utterance("Alpha beta gamma delta", 0, 7800, "en", words)])
        processor = make_processor(model, np.ones(16000 * 10, dtype=np.float32), "whisper")
        done = Future()
        done.set_result([
            {"start": 0, "end": 3800, "text": "Alpha beta",
             "emb": np.array([1.0, 0.0]), "ambiguous": False},
            {"start": 4000, "end": 7800, "text": "gamma delta",
             "emb": np.array([0.0, 1.0]), "ambiguous": False},
        ])
        # Retain the prior split contract only when explicitly testing the
        # legacy mode. Content-first mode preserves the original ASR turn.
        with (patch("sessions.processor.submit_speaker_analysis", return_value=done),
              patch.dict("os.environ", {"MEETING_CONTENT_FIRST_FUSION": "0"})):
            self.assertTrue(processor._cycle(final=True))
        self.assertEqual([item["text"] for item in processor.segments],
                         ["Alpha beta", "gamma delta"])
        self.assertEqual(processor.segments[0]["start_sample"], 0)
        self.assertEqual(processor.segments[1]["start_sample"], 64000)
        self.assertNotEqual(processor.segments[0]["speaker_id"], processor.segments[1]["speaker_id"])

    def test_translation_does_not_block_recognition(self):
        model = MultilingualModel([Utterance("Bonjour", 0, 800, "fr")])
        processor = make_processor(model, np.ones(16000 * 12, dtype=np.float32), "whisper")
        processor.translate_target = "en"
        pending = Future()
        with patch("sessions.processor.submit_translation", return_value=pending):
            self.assertTrue(processor._cycle(final=False))
            self.assertEqual(processor.segments[0]["text"], "Bonjour")
            self.assertFalse(pending.done())
        pending.set_result({1: "Hello"})
        processor._collect_translations()
        self.assertEqual(processor.segments[0]["translation"], "Hello")

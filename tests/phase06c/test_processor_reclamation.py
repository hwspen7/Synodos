"""Processor only reclaims confirmed silence and never arbitrary ASR failures."""
import asyncio
import unittest
from unittest.mock import patch
import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.contracts import Recognition


class EmptySpeechModel:
    def __init__(self, speech_detected):
        self.speech_detected = speech_detected

    def recognize_multilingual(self, audio, **kwargs):
        return Recognition(engine="whisper", speech_detected=self.speech_detected)


class ProcessorReclamationTests(unittest.TestCase):
    @staticmethod
    def processor(speech_detected, waveform):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate, max_seconds=30)
        buffer.append(waveform)
        with patch.dict("os.environ", {
            "MEETING_ASR_ENGINE": "whisper",
            "MEETING_SAFE_SILENCE_RECLAIM": "1",
            "MEETING_DIARIZATION_MODE": "cluster",
        }):
            instance = SessionProcessor(
                session_id="phase-06c-silence-test",
                buffer=buffer,
                settings=settings,
                models=EmptySpeechModel(speech_detected),
                outbox=asyncio.Queue(),
            )
        return instance

    def test_no_acoustic_speech_reclaims_pure_silence(self):
        processor = self.processor(False, np.zeros(16000*12, dtype=np.float32))
        with patch.dict("os.environ", {"MEETING_SAFE_SILENCE_RECLAIM": "1"}):
            processor._cycle(final=False)
        self.assertGreater(processor.buffer.start_offset(), 16000*8)
        self.assertEqual(processor.buffer.total_samples(), 16000*12 - processor.buffer.start_offset())
        self.assertEqual(processor.buffer.absolute_end_samples(), 16000*12)

    def test_untranscribed_detected_voice_must_be_preserved(self):
        processor = self.processor(True, np.zeros(16000*12, dtype=np.float32))
        processor._cycle(final=False)
        self.assertEqual(processor.buffer.start_offset(), 0)

    def test_ambiguous_vad_evidence_is_not_silence(self):
        processor = self.processor(None, np.zeros(16000*12, dtype=np.float32))
        processor._cycle(final=False)
        self.assertEqual(processor.buffer.start_offset(), 0)

    def test_voice_like_energy_does_not_get_dropped(self):
        samples = np.zeros(16000*12, dtype=np.float32)
        samples[16000:16000*3] = .1
        processor = self.processor(False, samples)
        processor._cycle(final=False)
        self.assertLessEqual(processor.buffer.start_offset(), 16000)

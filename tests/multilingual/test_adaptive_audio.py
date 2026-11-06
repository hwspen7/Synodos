"""Audio conditioning preserves time, protected samples and raw evidence."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.adaptive_audio import AdaptiveAudio
from speech.config import SpeechSettings
from speech.contracts import Recognition


RATE = 16000


def tone(hz: int, amplitude: float, seconds: float) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * time)).astype(np.float32)


class AdaptiveAudioTests(unittest.TestCase):
    def test_clean_voice_silence_and_clipping_pass_through(self):
        audio = np.concatenate((np.zeros(RATE, dtype=np.float32),
                                tone(220, 0.2, 1.0),
                                np.ones(RATE // 50, dtype=np.float32)))
        original = audio.copy()
        output = AdaptiveAudio(RATE).prepare(0, audio)
        np.testing.assert_array_equal(output, original)
        np.testing.assert_array_equal(audio, original)

    def test_steady_background_is_attenuated_without_losing_voiced_band(self):
        time = np.arange(2 * RATE) / RATE
        voice = tone(400, 0.15, 2)
        voice[:RATE] = 0
        noise = tone(50, 0.025, 2)
        audio = voice + noise
        output = AdaptiveAudio(RATE).prepare(0, audio)
        self.assertEqual(len(output), len(audio))
        self.assertLess(np.linalg.norm(output[RATE // 2:RATE]),
                        0.8 * np.linalg.norm(noise[RATE // 2:RATE]))
        for hz, minimum, maximum in [(50, 0.0, 0.85), (400, 0.99, 1.01)]:
            with self.subTest(hz=hz):
                probe = np.sin(2 * np.pi * hz * time[RATE:])
                ratio = np.dot(output[RATE:], probe) / np.dot(audio[RATE:], probe)
                self.assertGreater(ratio, minimum)
                self.assertLess(ratio, maximum)

    def test_quiet_voice_gain_stays_bounded_and_leaves_short_tail(self):
        quiet = tone(240, 0.010, 0.6)
        tail = np.array([0.01, -0.01, 0.01], dtype=np.float32)
        audio = np.concatenate((np.zeros(RATE // 2, dtype=np.float32), quiet, tail))
        output = AdaptiveAudio(RATE).prepare(0, audio)
        self.assertGreater(np.linalg.norm(output[-3203:-3]),
                           1.5 * np.linalg.norm(quiet[-3200:]))
        self.assertLessEqual(np.max(np.abs(output)), 0.02)
        np.testing.assert_array_equal(output[-len(tail):], tail)

    def test_broadband_noise_does_not_trigger_gain(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.015, RATE * 2).astype(np.float32)
        audio = np.concatenate((np.zeros(RATE // 2, dtype=np.float32), noise))
        output = AdaptiveAudio(RATE).prepare(0, audio)
        self.assertLess(np.linalg.norm(output[-RATE:]), 0.87 * np.linalg.norm(audio[-RATE:]))

    def test_normal_level_phrase_does_not_boost_its_quiet_tail(self):
        audio = np.concatenate((np.zeros(RATE // 2, dtype=np.float32),
                                tone(250, 0.20, 0.6), tone(250, 0.010, 0.4)))
        np.testing.assert_array_equal(AdaptiveAudio(RATE).prepare(0, audio), audio)

    def test_sliding_snapshots_reuse_samples_and_reset_after_a_gap(self):
        audio = tone(50, 0.025, 2)
        stream = AdaptiveAudio(RATE)
        preview = stream.prepare(500, audio[:RATE + 13])
        with patch.object(stream, "_condition", wraps=stream._condition) as condition:
            revised = stream.prepare(500 + 817, audio[817:])
            self.assertEqual(condition.call_count, RATE // 320)
        full = AdaptiveAudio(RATE).prepare(500, audio)
        np.testing.assert_array_equal(preview[:RATE], full[:RATE])
        np.testing.assert_array_equal(revised, full[817:])
        silence = np.zeros(RATE // 2, dtype=np.float32)
        np.testing.assert_array_equal(stream.prepare(100_000, silence), silence)

    def test_session_conditions_asr_but_preserves_raw_acoustic_input(self):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate)
        models = Mock()
        models.recognize_multilingual.return_value = Recognition((), "en")
        with patch("sessions.processor.load_settings", return_value=SpeechSettings(adaptive_audio=True)):
            session = SessionProcessor("adaptive-audio", buffer, settings, models, asyncio.Queue())
        session._schedule_acoustic = Mock()
        session._put = Mock()
        audio = tone(50, 0.025, 2)
        buffer.append(audio)
        session._cycle()

        offset, source = session._schedule_acoustic.call_args.args
        conditioned = models.recognize_multilingual.call_args.args[0]
        self.assertEqual(offset, 0)
        np.testing.assert_array_equal(source, audio)
        np.testing.assert_array_equal(buffer.snapshot()[1], audio)
        self.assertEqual(len(conditioned), len(audio))
        self.assertLess(np.linalg.norm(conditioned[RATE:]), 0.8 * np.linalg.norm(audio[RATE:]))

    def test_session_bypass_and_legacy_denoise_take_precedence(self):
        for options in (
            {},
            {"adaptive_audio": True, "enable_denoise": True},
            {"adaptive_audio": True, "native_streaming": True},
            {"adaptive_audio": True, "engine": "legacy"},
        ):
            with self.subTest(options=options):
                settings = Settings()
                with patch("sessions.processor.load_settings", return_value=SpeechSettings(**options)):
                    session = SessionProcessor("bypass", AudioBuffer(settings.sample_rate),
                                               settings, Mock(), asyncio.Queue())
                self.assertIsNone(session._adaptive_audio)

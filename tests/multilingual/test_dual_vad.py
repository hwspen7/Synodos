"""Optional second-VAD regression tests; no model downloads or real microphone."""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from speech.audio.dual_vad import crosscheck_speech_regions
from speech.asr.whisper import WhisperEngine


@contextmanager
def fake_webrtc(*, speech: bool = True, first_frames: int | None = None):
    module = types.ModuleType("webrtcvad")

    class FakeVad:
        def __init__(self, level):
            self.level = level
            self.calls = 0

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            assert sample_rate == 16000
            assert len(frame) == 960
            self.calls += 1
            return speech and (first_frames is None or self.calls <= first_frames)

    module.Vad = FakeVad
    with patch.dict(sys.modules, {"webrtcvad": module}):
        yield


@contextmanager
def fake_silero(regions):
    vad = types.ModuleType("faster_whisper.vad")

    class FakeVadOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vad.VadOptions = FakeVadOptions
    vad.get_speech_timestamps = lambda *args, **kwargs: regions
    main = types.ModuleType("faster_whisper")
    main.vad = vad
    with patch.dict(sys.modules, {"faster_whisper": main, "faster_whisper.vad": vad}):
        yield


class DualVADTests(unittest.TestCase):
    def test_false_webrtc_vote_rejects_silero_region(self):
        with fake_webrtc(speech=False):
            got = crosscheck_speech_regions(np.zeros(16000, np.float32), [{"start": 0, "end": 16000}])
        self.assertEqual(got, [])

    def test_both_detectors_support_speech(self):
        with fake_webrtc():
            got = crosscheck_speech_regions(np.ones(16000, np.float32) * .1, [{"start": 0, "end": 16000}])
        self.assertEqual(got, [{"start": 0, "end": 16000}])

    def test_very_short_webrtc_speech_rejected(self):
        with fake_webrtc(first_frames=3):
            got = crosscheck_speech_regions(np.ones(16000, np.float32) * .1, [{"start": 0, "end": 16000}])
        self.assertEqual(got, [])

    def test_very_low_fraction_is_rejected(self):
        with fake_webrtc(first_frames=10):
            got = crosscheck_speech_regions(np.ones(16000 * 8, np.float32) * .1, [{"start": 0, "end": 16000 * 8}])
        self.assertEqual(got, [])

    def test_does_not_create_speech_without_silero(self):
        with fake_webrtc():
            got = crosscheck_speech_regions(np.ones(16000, np.float32), [])
        self.assertEqual(got, [])

    def test_silero_regions_are_clipped_to_audio_length(self):
        with fake_webrtc():
            got = crosscheck_speech_regions(np.ones(16000, np.float32), [{"start": -100, "end": 1000000}])
        self.assertEqual(got, [{"start": 0, "end": 16000}])

    def test_rejects_wrong_sample_rate_and_bad_signal(self):
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            crosscheck_speech_regions(np.ones(100, np.float32), [], sample_rate=8000)
        with fake_webrtc():
            with self.assertRaisesRegex(ValueError, "finite mono"):
                crosscheck_speech_regions(np.array([float('nan')] * 600, dtype=np.float32), [])

    def test_missing_optional_dependency_has_instructions(self):
        with patch.dict(sys.modules, {"webrtcvad": None}):
            with self.assertRaisesRegex(RuntimeError, "webrtcvad-wheels"):
                crosscheck_speech_regions(np.ones(16000, np.float32), [])

    def test_real_engine_skips_whisper_without_two_votes(self):
        engine = WhisperEngine("dummy")
        fake = Mock()
        engine._model = fake
        with fake_silero([{"start": 0, "end": 16000}]), fake_webrtc(speech=False), \
                patch.dict(os.environ, {"MEETING_WHISPER_DUAL_VAD": "1"}):
            got = engine.recognize(np.zeros(16000, np.float32))
        self.assertEqual(got.utterances, ())
        fake.transcribe.assert_not_called()

    def test_real_engine_transcribes_if_two_detectors_agree(self):
        engine = WhisperEngine("dummy")
        fake = Mock()
        fake.transcribe.return_value = (
            iter([SimpleNamespace(start=.1, end=.9, text=" Hello", words=[],
                                  no_speech_prob=.1, avg_logprob=-.2)]),
            SimpleNamespace(language="en", language_probability=.99),
        )
        engine._model = fake
        with fake_silero([{"start": 0, "end": 16000}]), fake_webrtc(), \
                patch.dict(os.environ, {"MEETING_WHISPER_DUAL_VAD": "1"}):
            got = engine.recognize(np.ones(16000, np.float32) * .1)
        self.assertEqual([x.text for x in got.utterances], ["Hello"])
        self.assertEqual(got.language, "en")
        fake.transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()

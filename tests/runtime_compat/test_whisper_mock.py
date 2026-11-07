"""Mocked Whisper backend tests for automatic speech-region language tagging."""
from __future__ import annotations
import os
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from speech.asr.whisper import WhisperEngine


class WhisperAutoLanguageTests(unittest.TestCase):
    @staticmethod
    def fake_vad(regions):
        vad = types.ModuleType("faster_whisper.vad")
        vad.VadOptions = lambda **kw: types.SimpleNamespace(**kw)
        vad.get_speech_timestamps = lambda *_a, **_kw: regions
        parent = types.ModuleType("faster_whisper")
        parent.vad = vad
        return {"faster_whisper": parent, "faster_whisper.vad": vad}

    def test_auto_language_is_per_audio_window(self):
        sr = 16000
        class FakeModel:
            calls = 0
            def transcribe(self, arr, **kwargs):
                language = ("zh", "en")[self.calls]
                self.calls += 1
                info = types.SimpleNamespace(language=language, language_probability=.92)
                seg = types.SimpleNamespace(text="spoken", start=.05,
                                            end=len(arr)/sr-.05, words=[],
                                            no_speech_prob=.01, avg_logprob=-.01,
                                            compression_ratio=1)
                return iter([seg]), info
        engine = WhisperEngine("small")
        engine._model = FakeModel()
        regions = [{"start":sr, "end":sr*3}, {"start":sr*7, "end":sr*9}]
        with patch.dict(sys.modules, self.fake_vad(regions)):
            with patch.dict(os.environ, {"MEETING_WHISPER_WINDOW_LID": "1", "MEETING_WHISPER_DUAL_VAD": "0"}):
                result = engine.recognize(np.zeros(sr*10, dtype=np.float32))
        self.assertEqual(result.language, "mul")
        self.assertEqual([x.language for x in result.utterances], ["zh", "en"])
        self.assertEqual(engine._model.calls, 2)
        self.assertTrue(result.speech_detected)

    def test_no_vad_speech_cannot_be_reinterpreted_as_text(self):
        engine = WhisperEngine("small")
        with patch.dict(sys.modules, self.fake_vad([])):
            with patch.dict(os.environ, {"MEETING_WHISPER_DUAL_VAD": "0"}):
                result = engine.recognize(np.zeros(16000*8, dtype=np.float32))
        self.assertFalse(result.speech_detected)
        self.assertEqual(result.utterances, ())

    def test_fixed_language_can_still_be_explicit(self):
        class FakeModel:
            def transcribe(self, arr, **kw):
                self.kwargs = kw
                return iter([]), types.SimpleNamespace(language="pt", language_probability=.9)
        engine = WhisperEngine("small")
        engine._model = FakeModel()
        with patch.dict(sys.modules, self.fake_vad([{"start":0, "end":16000*3}])):
            with patch.dict(os.environ, {"MEETING_WHISPER_DUAL_VAD": "0"}):
                result = engine.recognize(np.zeros(16000*3, dtype=np.float32), language="en")
        self.assertEqual(result.language, "en")
        self.assertEqual(engine._model.kwargs["language"], "en")

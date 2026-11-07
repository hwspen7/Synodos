import sys
import types
from contextlib import contextmanager
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from speech.asr.whisper import WhisperEngine
from speech.asr.engine import recognize
from speech.config import SpeechSettings
from speech.contracts import Recognition
from speech.captions.translation import translate, TranslationUnavailable
from speech.diarization.overlap import OverlapDetector, OverlapUnavailable

@contextmanager
def mock_voiced_audio():
    """Supply deterministic speech regions without loading any VAD model."""
    vad = types.ModuleType("faster_whisper.vad")

    class FakeVadOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vad.VadOptions = FakeVadOptions
    vad.get_speech_timestamps = lambda samples, options, sampling_rate=16000: [
        {"start": 0, "end": len(samples)}
    ] if len(samples) else []
    main = types.ModuleType("faster_whisper")
    main.vad = vad
    with patch.dict(sys.modules, {"faster_whisper": main, "faster_whisper.vad": vad}):
        yield


class AdapterTests(unittest.TestCase):
    def test_whisper_multilingual_timestamp_and_hotword(self):
        engine = WhisperEngine("small")
        sample = SimpleNamespace(text=" hello", start=0, end=1.2,
                                 words=[SimpleNamespace(word="hello", start=.1, end=1.1)])
        fake = Mock()
        fake.transcribe.return_value = (iter([sample]), SimpleNamespace(language="en", language_probability=.99))
        engine._model = fake
        with mock_voiced_audio():
            result = engine.recognize(np.zeros(16000, np.float32), "auto", "MCP")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.utterances[0].timestamp_quality, "word")
        self.assertTrue(fake.transcribe.call_args.kwargs["multilingual"])
        self.assertEqual(fake.transcribe.call_args.kwargs["hotwords"], "MCP")

    def test_whisper_explicit_language(self):
        engine = WhisperEngine("small")
        fake = Mock()
        fake.transcribe.return_value = (iter([]), SimpleNamespace(language="fr", language_probability=.98))
        engine._model = fake
        with mock_voiced_audio():
            engine.recognize(np.zeros(1600, np.float32), language="fr")
        self.assertEqual(fake.transcribe.call_args.kwargs["language"], "fr")

    def test_live_hybrid_registry_uses_full_context_v2(self):
        from speech.asr.engine import _ENGINES
        whisper = Mock()
        whisper.recognize.return_value = Recognition(
            engine="whisper", speech_detected=False
        )
        qwen = Mock()
        with patch.dict(_ENGINES, {}, clear=True), patch(
            "speech.asr.whisper.WhisperEngine", return_value=whisper
        ), patch(
            "speech.asr.qwen_bridge.QwenWorkerClient", return_value=qwen
        ):
            settings = SpeechSettings(
                engine="hybrid", hybrid_qwen_python=r"E:\AI-Cache\qwen\python.exe"
            )
            answer = recognize(
                np.zeros(16000, dtype=np.float32),
                config=settings, language="auto"
            )
            self.assertEqual(answer.engine, "hybrid")
            self.assertEqual(len(_ENGINES), 1)
            backend = next(iter(_ENGINES.values()))
            self.assertTrue(backend.full_language_window)
            self.assertIs(backend.qwen, qwen)

    def test_legacy_fallback(self):
        res = recognize(np.zeros(1000, np.float32), config=SpeechSettings(engine="legacy"),
                        legacy=lambda audio, hotword: [{"start": 0, "end": 500, "text": "hello"}])
        self.assertEqual(res.utterances[0].text, "hello")

    def test_translation_missing_model_reports_unavailable(self):
        with self.assertRaises(TranslationUnavailable):
            translate("hello", "xx", "yy")

    def test_overlap_not_fabricated(self):
        with self.assertRaises(OverlapUnavailable):
            OverlapDetector().detect(np.zeros(16000, np.float32))

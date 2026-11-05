"""Deterministic hybrid routing, failover and timestamp contract tests."""
import unittest
from unittest.mock import Mock
import numpy as np
from speech.config import SpeechSettings
from speech.contracts import Recognition, Utterance, Word
from speech.hybrid import (
    HybridEngine, preferred_backend, WHISPER_LANGUAGES, QWEN_LANGUAGES,
)

def voiced():
    return np.full(32000, .05, dtype=np.float32)

def sample(text, language, start=100, end=900, confidence=.9):
    return Recognition(
        utterances=(Utterance(text, start, end, language,
                              language_confidence=confidence),) if text else (),
        language=language, engine="fake", speech_detected=True)

class HybridRoutingTests(unittest.TestCase):
    def test_ten_language_disjoint_policy(self):
        self.assertEqual(WHISPER_LANGUAGES, {"en", "zh", "ar", "id"})
        self.assertEqual(QWEN_LANGUAGES, {"fr", "es", "ja", "ru", "de", "hi"})
        for code in WHISPER_LANGUAGES:
            self.assertEqual(preferred_backend(code), "whisper")
        for code in QWEN_LANGUAGES:
            self.assertEqual(preferred_backend(code), "qwen3")

    def test_known_language_uses_one_preferred_engine(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.return_value = sample("hello", "en")
        qwen.recognize.return_value = sample("bonjour", "fr")
        self.assertEqual(HybridEngine(whisper, qwen).recognize(voiced(), "en")
                         .utterances[0].text, "hello")
        qwen.recognize.assert_not_called()
        whisper.recognize.reset_mock()
        self.assertEqual(HybridEngine(whisper, qwen).recognize(voiced(), "fr")
                         .utterances[0].text, "bonjour")
        whisper.recognize.assert_not_called()

    def test_known_fallback_on_worker_error(self):
        whisper, qwen = Mock(), Mock()
        qwen.recognize.side_effect = RuntimeError("worker offline")
        whisper.recognize.return_value = sample("recovered", "hi")
        result = HybridEngine(whisper, qwen).recognize(voiced(), "hi")
        self.assertEqual(result.utterances[0].text, "recovered")
        self.assertTrue(result.recovery_diagnostics["hybrid"]["fallback_accepted"])

    def test_known_fallback_on_empty_speech(self):
        whisper, qwen = Mock(), Mock()
        qwen.recognize.return_value = sample("", "hi")
        whisper.recognize.return_value = sample("recovered", "hi")
        result = HybridEngine(whisper, qwen).recognize(voiced(), "hi")
        self.assertTrue(result.recovery_diagnostics["hybrid"]["fallback_attempted"])

    def test_silence_does_not_double_infer(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.return_value = Recognition(speech_detected=False)
        HybridEngine(whisper, qwen).recognize(
            np.zeros(16000, dtype=np.float32), "en")
        qwen.recognize.assert_not_called()

    def test_auto_language_window_routing_and_preservation(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.return_value = Recognition(
            utterances=(Utterance("english", 100, 900, "en"),
                        Utterance("french", 1100, 1850, "fr",
                                  language_confidence=.95)),
            language="mul", speech_detected=True)
        qwen.recognize.return_value = sample("francais", "fr", 160, 750)
        result = HybridEngine(whisper, qwen).recognize(voiced(), "auto")
        self.assertEqual([u.text for u in result.utterances],
                         ["english", "francais"])
        self.assertEqual(qwen.recognize.call_args.kwargs["language"], "fr")
        self.assertEqual(result.recovery_diagnostics["hybrid"]["qwen_accepted"], 1)

    def test_auto_low_language_confidence_keeps_whisper(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.return_value = sample(
            "reference", "fr", confidence=.40)
        result = HybridEngine(whisper, qwen).recognize(voiced(), "auto")
        self.assertEqual(result.utterances[0].text, "reference")
        qwen.recognize.assert_not_called()

    def test_auto_uses_qwen_if_whisper_crashes(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.side_effect = RuntimeError("whisper unavailable")
        qwen.recognize.return_value = sample("backup", "fr")
        result = HybridEngine(whisper, qwen).recognize(voiced(), "auto")
        self.assertEqual(result.utterances[0].text, "backup")
        self.assertTrue(
            result.recovery_diagnostics["hybrid"]["whisper_failure_recovered"]
        )

    def test_prewarm_checks_both_backends(self):
        whisper, qwen = Mock(), Mock()
        HybridEngine(whisper, qwen)._get_model()
        whisper._get_model.assert_called()
        qwen.prewarm.assert_called_once()

    def test_auto_qwen_error_keeps_whisper(self):
        whisper, qwen = Mock(), Mock()
        whisper.recognize.return_value = sample("baseline", "ru")
        qwen.recognize.side_effect = RuntimeError("timeout")
        result = HybridEngine(whisper, qwen).recognize(voiced(), "auto")
        self.assertEqual(result.utterances[0].text, "baseline")

    def test_audio_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            HybridEngine(Mock(), Mock()).recognize(
                np.array([float("nan")], dtype=np.float32))

    def test_config_must_be_explicitly_opted_in(self):
        with self.assertRaises(ValueError):
            SpeechSettings(engine="hybrid").validate()
        self.assertEqual(
            SpeechSettings(engine="hybrid",
                           hybrid_qwen_python=r"E:\env\python.exe").validate().engine,
            "hybrid")
        self.assertEqual(SpeechSettings().engine, "whisper")

if __name__ == "__main__":
    unittest.main()

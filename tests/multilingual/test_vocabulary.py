"""Keep multiword hotwords intact and verify prompt-driven retries."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sessions.processor import SessionProcessor
from speech.config import SpeechSettings
from speech.contracts import Recognition, Utterance
from speech.asr.engine import _ENGINES, recognize
from speech.asr.qwen import QwenEngine
from speech.asr.qwen_bridge import QwenWorkerClient
from speech.asr.qwen_worker import process_request
from speech.captions.vocabulary import prompt_terms, repeated_terms
from speech.asr.whisper import WhisperEngine
from storage.hotwords import HotwordStore
from tests.multilingual.test_adapters import mock_voiced_audio


class VocabularyTests(unittest.TestCase):
    def test_store_and_meeting_keep_exact_phrases_and_older_single_words(self):
        with TemporaryDirectory() as folder:
            store = HotwordStore(Path(folder))
            saved = store.create("Engineering", ["Spring  Boot, Java", "MCP", "Spring Boot"])
            meeting = SimpleNamespace(
                hotword_store=store, hotword_ids=[saved["id"]], _hotword_limit=200,
            )
            SessionProcessor.load_hotwords(meeting)
            self.assertEqual(saved["words"], ["Spring Boot", "Java", "MCP"])
            self.assertEqual(meeting.active_hotword_count, 3)
            self.assertEqual(meeting._hotword_str, "Spring Boot\nJava\nMCP")

    def test_prompt_budget_never_slices_a_term(self):
        long_term = "a" * 1495
        self.assertEqual(prompt_terms([long_term, "Spring Boot", "API"]), [long_term, "API"])

    def test_whisper_receives_a_delimited_phrase_list(self):
        engine = WhisperEngine("small")
        engine._model = Mock()
        engine._model.transcribe.return_value = (
            iter([]), SimpleNamespace(language="en", language_probability=.99),
        )
        with mock_voiced_audio():
            engine.recognize(np.zeros(16000, np.float32), "en", "Spring Boot\nJava")
        self.assertEqual(engine._model.transcribe.call_args.kwargs["hotwords"], "Spring Boot, Java")

    def test_hybrid_qwen_worker_delivers_context_to_model_and_alignment_retry(self):
        qwen = QwenEngine("model", aligner=True)
        qwen._model = Mock()
        qwen._model.transcribe.side_effect = [
            ValueError("alignment failed"),
            [SimpleNamespace(language="French", text="Spring Boot", time_stamps=[])],
        ]
        client = QwenWorkerClient(sys.executable)
        config = SpeechSettings(engine="hybrid", hybrid_qwen_python="python")
        with mock_voiced_audio(), patch.object(client, "_request", side_effect=lambda request:
                process_request(qwen, request)), patch.dict(_ENGINES, {}, clear=True), patch(
                    "speech.asr.qwen_bridge.QwenWorkerClient", return_value=client,
                ), patch("speech.asr.whisper.WhisperEngine"):
            with self.assertWarnsRegex(RuntimeWarning, "Forced alignment failed"):
                result = recognize(np.zeros(16000, np.float32), language="fr",
                                   terms="Spring Boot\nJava", config=config)
        self.assertEqual(result.utterances[0].text, "Spring Boot")
        self.assertEqual(len(qwen._model.transcribe.call_args_list), 2)
        for call in qwen._model.transcribe.call_args_list:
            self.assertIn('["Spring Boot", "Java"]', call.kwargs["context"])
        self.assertFalse(qwen._model.transcribe.call_args.kwargs["return_time_stamps"])

    def test_repeat_detection_accepts_punctuation_and_adjacent_captions(self):
        rows = (Utterance("Spring Boot, Spring Boot", 0, 700),
                Utterance("Spring Boot.", 800, 1200))
        self.assertEqual(repeated_terms(rows, ["Spring Boot"]), {"springboot"})
        separated = (rows[0], Utterance("Spring Boot.", 2500, 2700))
        self.assertFalse(repeated_terms(separated, ["Spring Boot"]))

    def test_repeat_retry_keeps_unrelated_speech_and_never_deletes_for_empty_retry(self):
        prompted = Recognition((Utterance("Start here.", 0, 900),
                                 Utterance("Spring Boot Spring Boot Spring Boot.", 1000, 3000)),
                                "en", engine="whisper")
        for unprompted, expected in [
            (Recognition((Utterance("Start here.", 0, 950),
                          Utterance("Spring Boot.", 1100, 2400)), "en"), True),
            (Recognition((Utterance("Spring Boot.", 1100, 2400),), "en"), False),
            (Recognition((), "en"), False),
            (prompted, False),
        ]:
            with self.subTest(expected=expected, utterances=unprompted.utterances):
                engine = Mock()
                engine.recognize.side_effect = [prompted, unprompted]
                audio = np.zeros(48000, np.float32)
                config = SpeechSettings(engine="whisper")
                with patch.dict(_ENGINES, {}, clear=True), patch(
                        "speech.asr.whisper.WhisperEngine", return_value=engine):
                    result = recognize(audio, language="en", terms="Spring Boot", config=config)
                self.assertEqual(engine.recognize.call_count, 2)
                self.assertEqual(engine.recognize.call_args.args[-1], "")
                self.assertEqual(result.recovery_diagnostics["hotword_repeat_retry"]["accepted"],
                                 expected)
                self.assertEqual(result.utterances,
                                 (unprompted if expected else prompted).utterances)

    def test_clean_term_uses_single_decode(self):
        engine = Mock()
        engine.recognize.return_value = Recognition(
            (Utterance("We use Spring Boot.", 0, 1500),), "en", engine="whisper",
        )
        with patch.dict(_ENGINES, {}, clear=True), patch(
                "speech.asr.whisper.WhisperEngine", return_value=engine):
            result = recognize(np.zeros(24000, np.float32), language="en",
                               terms="Spring Boot", config=SpeechSettings(engine="whisper"))
        engine.recognize.assert_called_once()
        self.assertIsNone(result.recovery_diagnostics)

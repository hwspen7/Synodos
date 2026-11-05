"""Quality-candidate controls remain opt-in and keep prior audio evidence.

These deterministic tests do not claim a real-audio DER improvement.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from speech.quality_profile import quality_candidate_enabled
from speech.whisper import WhisperEngine
from speech.coverage_retention import choose_safe_drop
from speech.contracts import Utterance
from tests.multilingual.test_adapters import mock_voiced_audio


class QualityCandidateTests(unittest.TestCase):
    def _engine(self):
        engine = WhisperEngine("small")
        model = Mock()
        model.transcribe.side_effect = lambda chunk, **kwargs: (
            iter([SimpleNamespace(
                start=0.05, end=0.8, text=" Hello",
                words=[SimpleNamespace(start=0.08, end=0.75, word=" Hello")],
            )]),
            SimpleNamespace(language="en", language_probability=0.95),
        )
        engine._model = model
        return engine, model

    def test_reference_profile_remains_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(quality_candidate_enabled())
            engine, model = self._engine()
            with mock_voiced_audio():
                result = engine.recognize(np.zeros(16000, dtype=np.float32))
            self.assertEqual([u.text for u in result.utterances], ["Hello"])
            self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])

    def test_candidate_disables_duplicate_inner_vad_only(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "1",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT": "0",
        }, clear=True):
            self.assertTrue(quality_candidate_enabled())
            engine, model = self._engine()
            with mock_voiced_audio(), patch(
                "speech.whisper.recover_missing_voice",
                side_effect=lambda _samples, _regions, primary, *_args, **_kwargs:
                (primary, {"recovery_attempted": 0, "recovery_confirmed": 0}),
            ) as recover:
                result = engine.recognize(np.zeros(16000, dtype=np.float32))
            self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])
            self.assertEqual([u.text for u in result.utterances], ["Hello"])
            recover.assert_called_once()

    def test_candidate_bypasses_duplicate_inner_vad_for_explicit_language(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "1",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
        }, clear=True):
            engine, model = self._engine()
            with mock_voiced_audio(), patch(
                "speech.whisper.recover_missing_voice",
                side_effect=lambda _samples, _regions, primary, *_args, **_kwargs:
                (primary, {"recovery_attempted": 0}),
            ):
                engine.recognize(np.zeros(16000, dtype=np.float32), language="en")
            self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    def test_candidate_recovery_uses_bounded_subsecond_windows(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "1",
            "MEETING_ASR_TRACE_STAGES": "1",
        }, clear=True):
            engine, model = self._engine()
            recorded = {}
            def recover(samples, regions, primary, transcribe, kwargs, **options):
                recorded.update(options)
                return primary, {
                    "recovery_attempted": 0,
                    "recovery_confirmed": 0,
                    "recovery_failed": 0,
                }
            with mock_voiced_audio(), patch(
                "speech.whisper.recover_missing_voice", side_effect=recover,
            ):
                engine.recognize(np.zeros(16000, dtype=np.float32), language="en")
        self.assertEqual(recorded["max_windows"], 3)
        self.assertEqual(recorded["min_uncovered_ms"], 350)
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    def test_coverage_retention_defers_drop_then_reports_exhaustion(self):
        attempts = {}
        params = {
            "proposed_drop_samples": 32000,
            "audio_samples": 48000,
            "sample_rate": 16000,
            "global_start_samples": 0,
            "speech_intervals_ms": ((0, 1000), (1300, 2600)),
            "utterances": (Utterance("Hello", 0, 1000),),
            "attempts": attempts,
        }
        first, gaps = choose_safe_drop(**params)
        self.assertLess(first, 32000)
        self.assertFalse(gaps)
        second, gaps = choose_safe_drop(**params)
        self.assertLess(second, 32000)
        self.assertFalse(gaps)
        third, gaps = choose_safe_drop(**params)
        self.assertEqual(third, 32000)
        self.assertTrue(gaps)
        self.assertEqual(gaps[0]["status"], "voice_detected_transcript_unavailable")

    def test_false_or_empty_candidate_flag_is_not_enabled(self):
        for value in ("", "0", "false", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"MEETING_ASR_QUALITY_CANDIDATE": value}
            ):
                self.assertFalse(quality_candidate_enabled())


if __name__ == "__main__":
    unittest.main()

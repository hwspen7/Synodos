"""Isolated VAD and gap-retry A/B controls without loading GPU models.

These tests prove each switch changes only its intended Whisper stage.
They do NOT establish that transcript word error rate or DER improved.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from speech.whisper import WhisperEngine
from tests.multilingual.test_adapters import mock_voiced_audio


def fake_model():
    engine = WhisperEngine("small")
    model = Mock()

    def transcribe(samples, **kwargs):
        return iter([SimpleNamespace(
            start=0.10, end=0.85, text=" Original words",
            words=[SimpleNamespace(
                start=0.12, end=0.83, word=" Original words",
            )],
        )]), SimpleNamespace(language="en", language_probability=0.98)

    model.transcribe.side_effect = transcribe
    engine._model = model
    return engine, model


class WindowVadAblationTests(unittest.TestCase):
    def test_default_unchanged_for_windowed_auto_language(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "0",
            "MEETING_WHISPER_WINDOW_VAD_BYPASS": "0",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
            "MEETING_ASR_TRACE_STAGES": "1",
        }, clear=True):
            engine, model = fake_model()
            with mock_voiced_audio():
                result = engine.recognize(np.zeros(16000, np.float32))
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(len(result.utterances), 1)
        self.assertFalse(
            result.recovery_diagnostics["stage_trace"][
                "window_internal_vad_bypassed"
            ],
        )

    def test_opt_in_bypasses_only_inner_vad_not_outer_silero(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "0",
            "MEETING_WHISPER_WINDOW_VAD_BYPASS": "1",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
            "MEETING_ASR_TRACE_STAGES": "1",
        }, clear=True):
            engine, model = fake_model()
            with mock_voiced_audio():
                result = engine.recognize(np.zeros(16000, np.float32))
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual([u.text for u in result.utterances], ["Original words"])
        trace = result.recovery_diagnostics["stage_trace"]
        self.assertTrue(trace["window_internal_vad_bypassed"])
        self.assertTrue(trace["vad_intervals_ms"])
        self.assertEqual(trace["recovery_attempted"], 0)

    def test_explicit_language_still_uses_inner_vad(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "0",
            "MEETING_WHISPER_WINDOW_VAD_BYPASS": "1",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
        }, clear=True):
            engine, model = fake_model()
            with mock_voiced_audio():
                engine.recognize(np.zeros(16000, np.float32), language="en")
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])

    def test_independent_recovery_flag_does_not_enable_other_quality_candidates(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_QUALITY_CANDIDATE": "0",
            "MEETING_WHISPER_WINDOW_VAD_BYPASS": "0",
            "MEETING_WHISPER_GAP_RECOVERY": "1",
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT": "0",
            "MEETING_ASR_TRACE_STAGES": "1",
        }, clear=True):
            engine, model = fake_model()
            seen = []
            def recover(samples, regions, primary, transcribe, kwargs, **opts):
                seen.append((len(samples), kwargs["vad_filter"], len(primary)))
                return primary, {
                    "recovery_attempted": 1,
                    "recovery_confirmed": 0,
                    "recovery_failed": 0,
                }
            with mock_voiced_audio(), patch(
                "speech.whisper.recover_missing_voice", side_effect=recover,
            ):
                result = engine.recognize(np.zeros(16000, np.float32))
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(seen, [(16000, True, 1)])
        trace = result.recovery_diagnostics["stage_trace"]
        self.assertEqual(trace["recovery_attempted"], 1)
        self.assertEqual(trace["recovery_confirmed"], 0)


if __name__ == "__main__":
    unittest.main()

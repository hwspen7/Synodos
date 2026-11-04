"""Validate Whisper speech-region verification without downloading model weights."""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from speech.whisper import WhisperEngine


class _VadOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Model:
    def __init__(self, segments):
        self.segments = segments
        self.calls = 0
        self.kwargs = None

    def transcribe(self, samples, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return iter(self.segments), SimpleNamespace(language="en", language_probability=0.95)


def _segment(start: float, end: float, text: str):
    return SimpleNamespace(start=start, end=end, text=text, words=[])


class WhisperQualityGuardTests(unittest.TestCase):
    @staticmethod
    def _run(regions, pieces, *, flags=None):
        vad = types.ModuleType("faster_whisper.vad")
        vad.VadOptions = _VadOptions
        vad.get_speech_timestamps = lambda *args, **kwargs: regions
        main_module = types.ModuleType("faster_whisper")
        main_module.vad = vad
        model = _Model(pieces)
        engine = WhisperEngine("dummy", device="cpu", compute_type="int8")
        baseline_flags = {
            "MEETING_WHISPER_SKIP_INTERNAL_VAD": "0",
            "MEETING_WHISPER_GAP_RECOVERY": "0",
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT": "0",
        }
        baseline_flags.update(flags or {})
        with patch.dict("os.environ", baseline_flags):
            with patch.dict(sys.modules, {"faster_whisper": main_module, "faster_whisper.vad": vad}):
                with patch.object(engine, "_get_model", return_value=model):
                    output = engine.recognize(np.zeros(16000 * 4, dtype=np.float32))
        return output, model

    def test_complete_silence_does_not_start_whisper(self):
        output, model = self._run([], [_segment(0, 1, "Thanks for watching")])
        self.assertEqual(output.utterances, ())
        self.assertEqual(model.calls, 0)

    def test_text_outside_voiced_region_rejected(self):
        output, model = self._run([{"start": 0, "end": 16000}],
                                  [_segment(2, 3, "Thanks for watching")])
        self.assertEqual(output.utterances, ())
        # Baseline must never trigger an unrequested second decode.
        self.assertEqual(model.calls, 1)

    def test_real_speech_is_retained(self):
        output, model = self._run([{"start": 1600, "end": 19000}],
                                  [_segment(0.2, 1.1, "Meeting audio")])
        self.assertEqual([item.text for item in output.utterances], ["Meeting audio"])
        # Preserve the published V6 acoustic decoding behavior.
        self.assertTrue(model.kwargs["vad_filter"])
        self.assertEqual(
            model.kwargs["vad_parameters"]["min_silence_duration_ms"], 650,
        )
        self.assertFalse(model.kwargs["condition_on_previous_text"])
        self.assertEqual(model.kwargs["hallucination_silence_threshold"], 1.2)

    def test_second_decode_requires_explicit_opt_in(self):
        flags = {"MEETING_WHISPER_GAP_RECOVERY": "1"}
        output, model = self._run(
            [{"start": 0, "end": 16000}],
            [_segment(2, 3, "Unsupported")],
            flags=flags,
        )
        self.assertEqual(output.utterances, ())
        self.assertEqual(model.calls, 2)
        # Retry overrides only its own subwindow, not the normal ASR input.
        self.assertFalse(model.kwargs["vad_filter"])

    def test_vad_bypass_requires_explicit_opt_in(self):
        output, model = self._run(
            [{"start": 0, "end": 16000}],
            [_segment(0.1, 0.9, "Real words")],
            flags={"MEETING_WHISPER_SKIP_INTERNAL_VAD": "1"},
        )
        self.assertEqual(len(output.utterances), 1)
        self.assertFalse(model.kwargs["vad_filter"])

    def test_zero_length_fragment_is_rejected(self):
        output, _ = self._run([{"start": 0, "end": 16000}],
                               [_segment(0.9, 0.9, "Okay")])
        self.assertEqual(output.utterances, ())


if __name__ == "__main__":
    unittest.main()

"""Runtime invariants for buffer offsets, VAD evidence and Qwen windows."""
import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from speech.contracts import Recognition
from speech.qwen import QwenEngine


class BufferOffsetTests(unittest.TestCase):
    def test_end_timestamp_survives_buffer_cleanup(self):
        buffer = AudioBuffer(16000, max_seconds=30)
        buffer.append(np.zeros(16000 * 3, dtype=np.float32))
        self.assertEqual(buffer.total_samples(), 16000 * 3)
        self.assertEqual(buffer.absolute_end_samples(), 16000 * 3)
        buffer.drop_before(16000 * 2)
        self.assertEqual(buffer.total_samples(), 16000)
        self.assertEqual(buffer.absolute_end_samples(), 16000 * 3)
        self.assertEqual(buffer.start_offset(), 16000 * 2)
        buffer.append(np.zeros(16000, dtype=np.float32))
        self.assertEqual(buffer.total_samples(), 16000 * 2)
        self.assertEqual(buffer.absolute_end_samples(), 16000 * 4)

    def test_reclamation_maintains_global_audio_time(self):
        buffer = AudioBuffer(16000, max_seconds=30)
        buffer.append(np.zeros(16000 * 12, dtype=np.float32))
        from speech.silence_retention import safe_silence_prefix
        drop = safe_silence_prefix(buffer.snapshot()[1])
        buffer.drop_before(drop)
        self.assertEqual(buffer.total_samples(), 16000 * 12 - drop)
        self.assertEqual(buffer.absolute_end_samples(), 16000 * 12)


class QwenWindowTests(unittest.TestCase):
    def test_no_alignment_has_real_speech_window_boundaries(self):
        class Model:
            def transcribe(self, audio, language=None, return_time_stamps=False):
                return [types.SimpleNamespace(language="English", text="Test words", time_stamps=[])]

        vad = types.ModuleType("faster_whisper.vad")
        vad.VadOptions = lambda **_kw: object()
        vad.get_speech_timestamps = lambda _audio, _opts, sampling_rate=16000: [
            {"start": 16000, "end": 48000},
        ]
        parent = types.ModuleType("faster_whisper")
        parent.vad = vad
        engine = QwenEngine("Qwen3-ASR-0.6B", "cpu", aligner=False)
        engine._model = Model()
        with patch.dict(sys.modules, {"faster_whisper": parent, "faster_whisper.vad": vad}):
            with patch.dict(os.environ, {"MEETING_WHISPER_DUAL_VAD": "0"}):
                result = engine.recognize(np.zeros(16000 * 12, dtype=np.float32), "auto")
        self.assertEqual(len(result.utterances), 1)
        self.assertLess(result.utterances[0].end_ms, 5000)
        self.assertGreater(result.utterances[0].start_ms, 500)
        self.assertEqual(result.language, "en")
        self.assertTrue(result.speech_detected)

    def test_vad_absence_does_not_create_qwen_transcript(self):
        vad = types.ModuleType("faster_whisper.vad")
        vad.VadOptions = lambda **_kw: object()
        vad.get_speech_timestamps = lambda *_a, **_kw: []
        parent = types.ModuleType("faster_whisper")
        parent.vad = vad
        engine = QwenEngine("Qwen3-ASR-0.6B", "cpu", aligner=False)
        with patch.dict(sys.modules, {"faster_whisper": parent, "faster_whisper.vad": vad}):
            result = engine.recognize(np.zeros(16000 * 3, dtype=np.float32))
        self.assertEqual(result.utterances, ())
        self.assertFalse(result.speech_detected)


class VoiceEvidenceContractTests(unittest.TestCase):
    def test_unknown_vs_confirmed_speech_are_distinct(self):
        self.assertIsNone(Recognition().speech_detected)
        self.assertFalse(Recognition(speech_detected=False).speech_detected)
        self.assertTrue(Recognition(speech_detected=True).speech_detected)

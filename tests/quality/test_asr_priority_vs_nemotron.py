"""Independent acoustic notifications must not starve due microphone ASR."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


class AsrPriorityTests(unittest.TestCase):
    def make_processor(self):
        settings = Settings(
            vad_threshold_s=2.5,
            rec_threshold_s=1.0,
            asr_model="fake-asr",
            fsmn_vad_model="fake-vad",
            punc_model="fake-punctuation",
            spk_model="fake-speaker",
        )
        buffer = AudioBuffer(16000, max_seconds=30)
        buffer.append(np.ones(48000, dtype=np.float32) * 0.01)
        with patch.dict(os.environ, {
            "MEETING_DIARIZATION_DIAGNOSTICS": "0",
            "MEETING_ECAPA_SHADOW": "0",
        }):
            processor = SessionProcessor(
                "test-asr-priority", buffer, settings,
                object(), asyncio.Queue(),
            )
        processor._async_ready.set()
        processor._finalize = Mock()
        return processor

    def test_due_asr_wins_over_independent_acoustic_notification(self):
        processor = self.make_processor()

        def due_audio(final=False):
            processor.stop()
            return True

        processor._cycle = Mock(side_effect=due_audio)
        processor._collect_speakers = Mock()
        processor.run()
        processor._cycle.assert_called_once_with(final=False)
        processor._collect_speakers.assert_not_called()

    def test_completed_speaker_result_is_collected_without_new_pcm(self):
        processor = self.make_processor()
        processor._last_inferred_absolute_end = processor.buffer.absolute_end_samples()

        def completed_speaker():
            processor.stop()
            return True

        processor._collect_speakers = Mock(side_effect=completed_speaker)
        processor._collect_acoustic = Mock(return_value=False)
        processor._schedule_acoustic = Mock()
        processor.run()
        processor._collect_speakers.assert_called_once()


if __name__ == "__main__":
    unittest.main()

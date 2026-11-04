"""Prevent silent deletion of acoustically supported untranscribed speech."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.config import SpeechSettings
from speech.contracts import Recognition, Utterance
from speech.coverage_retention import choose_safe_drop


class CoverageFakeModel:
    def recognize_multilingual(self, audio, language, terms, config):
        # ASR missed the first 2s of genuine speech but heard later words.
        return Recognition(
            (Utterance("The later decision", 4000, 6000, "en"),),
            "en", speech_detected=True,
            speech_intervals_ms=((0, 2000), (4000, 6000)),
        )
    def embed(self, audio):
        return None


class SpeechCoverageRetentionTests(unittest.TestCase):
    def test_unrecognized_prefix_is_retained_for_two_retries_then_flagged(self):
        attempts = {}
        args = {
            "proposed_drop_samples": 6 * 16000,
            "audio_samples": 10 * 16000,
            "sample_rate": 16000,
            "global_start_samples": 0,
            "speech_intervals_ms": ((0, 2000), (4000, 6000)),
            "utterances": (Utterance("Later text", 4000, 6000),),
            "attempts": attempts,
        }
        first, gaps = choose_safe_drop(**args)
        self.assertEqual(first, 0)
        self.assertFalse(gaps)
        second, gaps = choose_safe_drop(**args)
        self.assertEqual(second, 0)
        self.assertFalse(gaps)
        third, gaps = choose_safe_drop(**args)
        self.assertEqual(third, 6 * 16000)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start_ms"], 0)
        self.assertEqual(gaps[0]["end_ms"], 2000)
        fourth, gaps = choose_safe_drop(**args)
        self.assertEqual(fourth, 6 * 16000)
        self.assertEqual(len(gaps), 1)

    def test_voice_with_text_is_not_held_and_final_warns_on_no_text(self):
        full, events = choose_safe_drop(
            proposed_drop_samples=4 * 16000,
            audio_samples=5 * 16000,
            sample_rate=16000,
            global_start_samples=0,
            speech_intervals_ms=((0, 4000),),
            utterances=(Utterance("All speech", 0, 4000),),
            attempts={},
        )
        self.assertEqual(full, 4 * 16000)
        self.assertEqual(events, [])
        remaining, events = choose_safe_drop(
            proposed_drop_samples=0,
            audio_samples=5 * 16000,
            sample_rate=16000,
            global_start_samples=10 * 16000,
            speech_intervals_ms=((500, 2100),),
            utterances=(),
            attempts={},
            final=True,
        )
        self.assertEqual(remaining, 0)
        self.assertEqual(events[0]["start_ms"], 10500)
        self.assertEqual(events[0]["status"],
                         "voice_detected_transcript_unavailable")

    def test_processor_does_not_drop_earlier_undecoded_voice_silently(self):
        settings = Settings()
        sr = settings.sample_rate
        buffer = AudioBuffer(sr, max_seconds=60)
        fake = CoverageFakeModel()
        events = []
        with patch("sessions.processor.load_settings",
                   return_value=SpeechSettings(engine="whisper")):
            processor = SessionProcessor(
                "voice-gap", buffer, settings, fake, asyncio.Queue()
            )
        processor._put = events.append
        processor._schedule_acoustic = lambda *args: None
        processor._collect_speakers = lambda *args, **kw: False
        processor._collect_acoustic = lambda *args, **kw: False
        processor._collect_translations = lambda *args, **kw: None
        processor._recluster = lambda *args: None
        with patch("sessions.processor.submit_speaker_analysis", return_value=None):
            for i in range(3):
                buffer.append(np.ones(sr * (10 if i == 0 else 2),
                                      dtype=np.float32) * 0.03)
                processor._cycle(final=False)
                if i < 2:
                    self.assertEqual(buffer.start_offset(), 0)
                else:
                    self.assertEqual(buffer.start_offset(), 6 * sr)
        self.assertEqual(processor.segments[0]["text"], "The later decision")
        self.assertEqual(len(processor._speech_coverage_gaps), 1)
        self.assertEqual(
            processor.snapshot_state()["speech_coverage_gaps"][0]["start_ms"], 0
        )
        self.assertEqual(len([
            item for item in events if item["type"] == "speech_coverage"
        ]), 1)
        with patch("sessions.processor.load_settings",
                   return_value=SpeechSettings(engine="whisper")):
            other = SessionProcessor(
                "voice-gap-resume", AudioBuffer(sr), settings, fake,
                asyncio.Queue()
            )
        other.load_state(processor.snapshot_state())
        self.assertEqual(len(other._speech_coverage_gaps), 1)

    def test_legacy_provider_without_vad_contract_is_unchanged(self):
        end, gaps = choose_safe_drop(
            proposed_drop_samples=16000,
            audio_samples=16000,
            sample_rate=16000,
            global_start_samples=0,
            speech_intervals_ms=(),
            utterances=(),
            attempts={},
        )
        self.assertEqual(end, 16000)
        self.assertEqual(gaps, [])


if __name__ == "__main__":
    unittest.main()

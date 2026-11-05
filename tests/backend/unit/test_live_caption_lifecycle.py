"""Streaming caption contracts for live drafts and durable confirmations."""
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
from speech.streaming import PreviewTracker


class FakeMultilingualModels:
    def __init__(self):
        self.text = "The meeting is underway."

    def recognize_multilingual(self, audio, language, terms, config):
        return Recognition(
            utterances=(Utterance(self.text, 0, 2000, "en"),),
            language="en", speech_detected=True,
        )

    def embed(self, audio):
        return None


class RealtimeSubtitleContracts(unittest.TestCase):
    def test_previews_have_monotonic_revision_even_on_clear(self):
        tracker = PreviewTracker()
        first = tracker.update("First words", "en", 1000, 2500)
        self.assertEqual(first.revision, 1)
        self.assertIsNone(tracker.update("First words", "en", 1000, 2500))
        changed = tracker.update("First words continue", "en", 1000, 3200)
        self.assertEqual(changed.revision, 2)
        cleared = tracker.clear(1000, 3800)
        self.assertIsNotNone(cleared)
        self.assertEqual(cleared.text, "")
        self.assertEqual(cleared.revision, 3)
        self.assertIsNone(tracker.clear(1000, 3800))
        later = tracker.update("Next turn", "en", 4000, 4800)
        self.assertEqual(later.revision, 4)

    def make_processor(self, engine="whisper"):
        settings = Settings()
        buffer = AudioBuffer(settings.sample_rate, max_seconds=60)
        models = FakeMultilingualModels()
        events = []
        speech = SpeechSettings(engine=engine, model="large-v3",
                                preview_interval_s=2.5,
                                confirmation_delay_s=1.0)
        with patch("sessions.processor.load_settings", return_value=speech):
            processor = SessionProcessor(
                "preview-contract", buffer, settings, models, asyncio.Queue(),
            )
        processor._put = events.append
        processor._schedule_acoustic = lambda *args: None
        processor._collect_speakers = lambda *args: False
        processor._collect_acoustic = lambda *args: False
        processor._collect_translations = lambda *args: None
        processor._recluster = lambda *args: None
        return processor, models, events

    def test_ghost_caption_on_digital_silence_never_reaches_speakers(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        processor.buffer.append(np.zeros(sr * 9, dtype=np.float32))
        models.text = "Hey, who are you?"
        with patch("sessions.processor.submit_speaker_analysis") as speaker:
            self.assertFalse(processor._cycle(final=False))
        self.assertFalse(processor.segments)
        self.assertEqual([x for x in events if x["type"] == "partial"], [])
        speaker.assert_not_called()
        self.assertLessEqual(processor.buffer.total_samples(), sr * 3)

    def test_ghost_caption_on_steady_microphone_noise_is_suppressed(self):
        processor, models, events = self.make_processor(engine="hybrid")
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.003, 9 * processor.settings.sample_rate)
        processor.buffer.append(noise.astype(np.float32))
        models.text = "You."
        with patch("sessions.processor.submit_speaker_analysis") as speaker:
            self.assertFalse(processor._cycle(final=False))
        self.assertEqual(processor.segments, [])
        self.assertFalse(any(x["type"] == "partial" for x in events))
        speaker.assert_not_called()

    def test_draft_before_four_seconds_not_confirmed_and_then_cleared_once(self):
        processor, models, events = self.make_processor()
        self.assertEqual(processor._process_threshold_s, 2.5)
        self.assertGreaterEqual(processor._min_commit_context_s, 7.0)

        sr = processor.settings.sample_rate
        processor.buffer.append(np.ones(int(sr * 2.6), dtype=np.float32) * 0.03)
        self.assertFalse(processor._cycle(final=False))
        drafts = [x for x in events if x["type"] == "partial"]
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["text"], models.text)
        self.assertEqual(drafts[0]["revision"], 1)
        self.assertEqual(len(processor.segments), 0)

        models.text = "The meeting is underway. We need a plan."
        processor.buffer.append(np.ones(int(sr * 1.6), dtype=np.float32) * 0.03)
        self.assertFalse(processor._cycle(final=False))
        drafts = [x for x in events if x["type"] == "partial"]
        self.assertEqual(len(drafts), 2)
        self.assertEqual(drafts[-1]["revision"], 2)
        self.assertEqual(len(processor.segments), 0)

        # Once a sufficiently old phrase is durably committed, the backend
        # issues exactly one explicit draft clear, not a revision-zero hack.
        processor.buffer.append(np.ones(int(sr * 3.0), dtype=np.float32) * 0.03)
        with patch("sessions.processor.submit_speaker_analysis", return_value=None):
            self.assertTrue(processor._cycle(final=False))
        drafts = [x for x in events if x["type"] == "partial"]
        self.assertEqual(len(drafts), 3)
        self.assertEqual(drafts[-1]["text"], "")
        self.assertEqual(drafts[-1]["revision"], 3)
        self.assertEqual(len(processor.segments), 1)
        self.assertEqual(processor.segments[0]["text"], models.text)
        self.assertTrue(any(x["type"] == "transcript" for x in events))

        # A duplicate or unchanged call must not clear the draft again.
        processor._cycle(final=False)
        self.assertEqual(len([x for x in events if x["type"] == "partial"]), 3)

    def test_live_hybrid_tail_echo_not_saved_as_new_speaker(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        prior_end = sr * 14
        processor.segments = [{
            "id": 1, "start_sample": prior_end - sr,
            "end_sample": prior_end,
            "text": "Who are you?", "speaker_id": 1,
            "speaker_name": "Speaker 1", "cluster": -3, "emb": None,
        }]
        processor._next_segment_id = 2
        processor.buffer.set_start_offset(prior_end)
        processor.buffer.append(
            np.full(int(sr * 7.5), .03, dtype=np.float32)
        )
        models.recognize_multilingual = lambda *a, **kw: Recognition(
            utterances=(Utterance("You.", 0, 250, "en"),),
            language="en", speech_detected=True,
        )
        processor._rebuild_and_push = lambda: False
        with patch("sessions.processor.submit_speaker_analysis") as speaker:
            self.assertFalse(processor._cycle(final=False))
        speaker.assert_not_called()
        self.assertEqual(len(processor.segments), 1)
        self.assertEqual(processor.segments[0]["text"], "Who are you?")
        self.assertFalse(any(
            row.get("type") == "partial" and row.get("text") == "You."
            for row in events
        ))

    def test_tail_echo_is_filtered_before_provisional_caption(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        processor.segments = [{
            "id": 1, "start_sample": sr, "end_sample": sr * 2,
            "text": "Hi, how are you man?", "speaker_id": 1,
            "speaker_name": "Speaker 1", "cluster": -3, "emb": None,
        }]
        processor._next_segment_id = 2
        processor.buffer.set_start_offset(sr * 2)
        processor.buffer.append(np.full(sr * 3, 0.03, dtype=np.float32))
        models.recognize_multilingual = lambda *a, **kw: Recognition(
            utterances=(Utterance(
                "Man.", 0, 930, "en", timestamp_quality="window_estimate"
            ),),
            language="en", speech_detected=True,
        )
        # The fixture already contains a published sentence. Suppress the
        # unrelated first transcript-snapshot refresh when checking the ASR
        # cycle's new-candidate return value.
        processor._rebuild_and_push = lambda: False
        with patch("sessions.processor.submit_speaker_analysis") as speaker:
            self.assertFalse(processor._cycle(final=False))
        self.assertEqual(len(processor.segments), 1)
        self.assertFalse(any(x["type"] == "partial" and x["text"]
                             for x in events))
        speaker.assert_not_called()

    def test_live_confirmed_acoustic_off_boundary_consumes_small_tail(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        processor.buffer.append(np.full(sr * 8, .03, dtype=np.float32))
        models.recognize_multilingual = lambda *a, **kw: Recognition(
            utterances=(Utterance(
                "Hi, how are you man?", 1000, 2000, "en"
            ),),
            language="en", speech_detected=True,
            speech_intervals_ms=((950, 2250),),
        )
        with patch("sessions.processor.submit_speaker_analysis",
                   return_value=None):
            self.assertTrue(processor._cycle(final=False))
        self.assertEqual(len(processor.segments), 1)
        self.assertEqual(processor.segments[0]["end_sample"], 2 * sr)
        self.assertEqual(processor.buffer.start_offset(), int(2.33 * sr))

    def test_two_inference_windows_do_not_decode_same_leftover_syllable(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        processor.buffer.append(np.full(sr * 8, .03, dtype=np.float32))
        calls = []

        def transcribe(audio, **kwargs):
            calls.append(processor.buffer.start_offset())
            if processor.buffer.start_offset() < int(sr * 2.70):
                return Recognition(
                    utterances=(Utterance(
                        "Hey, how are you, man?", 1000, 2000, "en"
                    ),),
                    language="en", speech_detected=True,
                    speech_intervals_ms=((900, 2750),),
                )
            return Recognition(engine="hybrid", speech_detected=False)

        models.recognize_multilingual = transcribe
        with patch("sessions.processor.submit_speaker_analysis",
                   return_value=None):
            processor._cycle(final=False)
            self.assertEqual(len(processor.segments), 1)
            self.assertGreaterEqual(
                processor.buffer.start_offset(), int(sr * 2.80)
            )
            processor.buffer.append(np.full(sr * 2, .03, dtype=np.float32))
            processor._cycle(final=False)
        self.assertEqual(len(processor.segments), 1)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[-1], int(sr * 2.70))

    def test_speech_island_with_pending_reply_not_trimmed(self):
        processor, models, events = self.make_processor(engine="hybrid")
        sr = processor.settings.sample_rate
        processor.buffer.append(np.full(sr * 8, .03, dtype=np.float32))
        models.recognize_multilingual = lambda *args, **kwargs: Recognition(
            utterances=(
                Utterance("I am here.", 1000, 2000, "en"),
                Utterance("Yes.", 2600, 7400, "en"),
            ), language="en", speech_detected=True,
            speech_intervals_ms=((900, 2750), (7000, 7700)),
        )
        with patch("sessions.processor.submit_speaker_analysis",
                   return_value=None):
            processor._cycle(final=False)
        self.assertEqual(len(processor.segments), 1)
        # Its own short, unfinished reply protects any overlapped audio
        # from the newest published cursor.
        self.assertLessEqual(processor.buffer.start_offset(), 2 * sr)

    def test_stale_preview_clears_even_when_vad_stays_positive(self):
        processor, models, events = self.make_processor(engine="whisper")
        sr = processor.settings.sample_rate
        processor.buffer.append(np.full(int(sr * 2.6), .03, dtype=np.float32))
        self.assertFalse(processor._cycle(final=False))
        self.assertEqual(len([e for e in events if e["type"] == "partial"]), 1)
        models.text = ""
        processor.buffer.append(np.full(int(sr * 0.8), .03, dtype=np.float32))
        self.assertFalse(processor._cycle(final=False))
        texts = [e["text"] for e in events if e["type"] == "partial"]
        self.assertEqual(texts, ["The meeting is underway.", ""])

    def test_finalization_clears_pending_draft_without_dropping_words(self):
        processor, models, events = self.make_processor()
        sr = processor.settings.sample_rate
        processor.buffer.append(np.ones(int(sr * 2.6), dtype=np.float32) * 0.03)
        processor._cycle(final=False)
        self.assertEqual(len(processor.segments), 0)
        with patch("sessions.processor.submit_speaker_analysis", return_value=None):
            processor._cycle(final=True)
        drafts = [x for x in events if x["type"] == "partial"]
        self.assertEqual([x["text"] for x in drafts], [models.text, ""])
        self.assertEqual(len(processor.segments), 1)
        self.assertEqual(processor.segments[0]["text"], models.text)


if __name__ == "__main__":
    unittest.main()

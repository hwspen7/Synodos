"""Translation stays separate from ASR, and persisted results own a target."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api import speech_features
from intelligence.model_gateway import AIModelError
from sessions.translation_tasks import CaptionTranslation
from speech.captions.corrections import correct_segment
from speech.captions.translation import TranslationUnavailable, translate_batch
from storage.meetings import MeetingStore


class TranslationProviderTests(unittest.TestCase):
    def test_mixed_sources_use_local_cpu_and_return_only_requested_unique_ids(self):
        calls = []

        def reply(config, path, body):
            calls.append((config, path, body))
            return {"message": {"content": json.dumps({"translations": [
                {"id": 4, "text": "Good morning."},
                {"id": 99, "text": "Unexpected."},
                {"id": 8, "text": "First."},
                {"id": 8, "text": "Duplicate."},
            ]})}}

        with patch.dict("os.environ", {"MEETING_AI_ENABLED": "1",
                        "MEETING_AI_ENDPOINT": "http://127.0.0.1:11435",
                        "MEETING_AI_PROVIDER": "ollama"}, clear=True), patch(
            "speech.captions.translation.translate",
            side_effect=["Already English.", TranslationUnavailable("Missing pair"),
                         TranslationUnavailable("Unknown language")],
        ), patch("intelligence.model_gateway._request", side_effect=reply):
            actual = translate_batch(((1, "Already English.", "en"),
                                      (4, "Bonjour.", "fr"), (8, "Hola.", "und")), "en")
        self.assertEqual(actual, {1: "Already English.", 4: "Good morning."})
        config, path, body = calls[0]
        self.assertEqual(path, "/api/chat")
        self.assertLessEqual(config.timeout_s, 45)
        self.assertEqual(body["options"]["num_gpu"], 0)
        prompt = json.loads(body["messages"][-1]["content"])
        self.assertEqual(prompt["target_language"], "English")
        self.assertEqual([row["id"] for row in prompt["captions"]], [4, 8])
        self.assertEqual(prompt["captions"][1]["source_language"], "Detect from text")

    def test_provider_outage_keeps_completed_offline_results(self):
        with patch.dict("os.environ", {"MEETING_AI_ENABLED": "1"}, clear=True), patch(
            "speech.captions.translation.translate",
            side_effect=["Hello", TranslationUnavailable("Missing pair")],
        ), patch("intelligence.model_gateway._request", side_effect=AIModelError("timeout")):
            actual = translate_batch(((1, "Bonjour", "fr"), (2, "Hola", "es")), "en")
        self.assertEqual(actual, {1: "Hello"})

    def test_remote_endpoint_is_not_used_for_automatic_translation(self):
        with patch.dict("os.environ", {"MEETING_AI_ENABLED": "1",
                        "MEETING_AI_ENDPOINT": "https://example.test",
                        "MEETING_AI_ALLOW_REMOTE": "1"}, clear=True), patch(
            "intelligence.model_gateway._request",
        ) as request:
            with self.assertRaises(TranslationUnavailable):
                translate_batch(((1, "Bonjour", "fr"),), "en")
        request.assert_not_called()


class SavedTranslationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MeetingStore(Path(directory.name))
        self.meeting = "translation-test"
        self.store.save_state({
            "id": self.meeting,
            "ui_settings": {"translate_to": "en"},
            "segments": [
                {"id": 1, "start_sample": 0, "end_sample": 16000, "speaker_id": 2,
                 "speaker_name": "Speaker 2", "text": "Bonjour.", "language": "fr"},
                {"id": 2, "start_sample": 16000, "end_sample": 32000, "speaker_id": 3,
                 "speaker_name": "Speaker 3", "text": "Already English.", "language": "en"},
            ],
        })

    def request(self):
        return asyncio.run(speech_features.translate_saved_meeting(
            self.meeting, speech_features.SavedTranslationRequest(target_language="en"),
        ))

    def test_backfill_persists_translation_and_preserves_caption(self):
        completed = Future()
        completed.set_result({1: "Hello."})
        with patch.object(speech_features, "meeting_store", self.store), patch.object(
            speech_features, "submit_translation", return_value=completed,
        ) as submit:
            self.assertEqual(self.request()["translated"], 1)
            self.assertEqual(self.request()["attempted"], 0)
        submit.assert_called_once()
        items = self.store.list_segments(self.meeting)
        self.assertEqual((items[0]["text"], items[0]["speaker_id"], items[0]["end_ms"]),
                         ("Bonjour.", 2, 1000))
        self.assertEqual((items[0]["translation"], items[0]["translation_target"]), ("Hello.", "en"))
        self.assertFalse(items[1]["translation"])
        self.assertIn("Hello.", self.store.export(self.meeting, "md"))
        state = self.store.load_state(self.meeting)
        correct_segment(state, 1, "Bonsoir.", 0)
        self.assertNotIn("translation", state["segments"][0])

    def test_saved_backfill_runs_real_queue_and_gateway_with_a_model_fixture(self):
        response = {"message": {"content": json.dumps({"translations": [
            {"id": 1, "text": "Hello."},
        ]})}}
        with patch.object(speech_features, "meeting_store", self.store), patch.dict(
            "os.environ", {"MEETING_AI_ENABLED": "1", "MEETING_AI_PROVIDER": "ollama"}, clear=True,
        ), patch("speech.captions.translation.translate", side_effect=TranslationUnavailable("Missing pair")), patch(
            "intelligence.model_gateway._request", return_value=response,
        ) as gateway:
            self.assertEqual(self.request()["translated"], 1)
        self.assertEqual(gateway.call_count, 1)
        self.assertEqual(self.store.list_segments(self.meeting)[0]["translation"], "Hello.")

    def test_backfill_allows_saved_stop_ack_but_rejects_active_recording(self):
        from fastapi import HTTPException

        completed = Future()
        completed.set_result({1: "Hello."})
        processor = SimpleNamespace(stopped=False, _final_save_ok=False)
        with patch.object(speech_features, "meeting_store", self.store), patch.dict(
            speech_features.meeting_sessions, {self.meeting: SimpleNamespace(processor=processor)},
        ), patch.object(speech_features, "submit_translation", return_value=completed) as submit:
            with self.assertRaises(HTTPException) as rejected:
                self.request()
            self.assertEqual(rejected.exception.status_code, 409)
            submit.assert_not_called()
            processor.stopped = processor._final_save_ok = True
            self.assertEqual(self.request()["translated"], 1)

    def test_backfill_discards_results_if_source_was_edited_in_flight(self):
        def finish(batch, target):
            self.assertEqual(batch, (CaptionTranslation(1, "Bonjour.", "fr"),))
            state = self.store.load_state(self.meeting)
            correct_segment(state, 1, "Bonsoir.", 0)
            self.store.save_state(state)
            completed = Future()
            completed.set_result({1: "Hello."})
            return completed

        with patch.object(speech_features, "meeting_store", self.store), patch.object(
            speech_features, "submit_translation", side_effect=finish,
        ):
            self.assertEqual(self.request()["translated"], 0)
        item = self.store.list_segments(self.meeting)[0]
        self.assertEqual(item["text"], "Bonsoir.")
        self.assertFalse(item["translation"])

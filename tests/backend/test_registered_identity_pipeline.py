"""Speaker recognition pipeline integration contracts without inference models."""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sessions.processor import SessionProcessor
from storage.speakers import SpeakerStore


def embedding(index=0):
    result = np.zeros(192, dtype=np.float32)
    result[index] = 1
    return result


class SpeakerSessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SpeakerStore(Path(self.temp.name))
        speaker = self.store.create("Registered Person")
        self.person_id = speaker["id"]
        self.store.add_vectors(self.person_id, [{
            "emb": embedding(), "duration_s": 6.0,
            "short": False, "embedding_model": "camplus",
            "source": "recording",
        }])

    def make_processor(self, participant_ids=None, auto_match_saved=True):
        # Exercise standalone configuration/matching methods without
        # allocating GPU models, audio workers or a real WebSocket.
        proc = object.__new__(SessionProcessor)
        proc.speaker_store = self.store
        proc.participant_ids = list(participant_ids or [])
        proc.auto_match_saved = auto_match_saved
        proc._participants = []
        proc._auto_profile_mode = False
        proc._auto_match_cache = {}
        proc.spk_threshold = 0.3
        return proc

    def test_empty_selection_matches_saved_library(self):
        with patch.dict(os.environ, {"MEETING_AUTO_SPEAKER_MATCH": "1"}):
            processor = self.make_processor()
            processor.load_participants()
            self.assertTrue(processor._auto_profile_mode)
            self.assertEqual(processor.participant_ids, [])
            who, value = processor._match_participant(embedding())
            self.assertEqual(who, self.person_id)
            self.assertGreater(value, 0.98)

    def test_user_opt_out_prevents_library_search(self):
        with patch.dict(os.environ, {"MEETING_AUTO_SPEAKER_MATCH": "1"}):
            processor = self.make_processor(auto_match_saved=False)
            processor.load_participants()
            self.assertFalse(processor._auto_profile_mode)
            self.assertEqual(processor._match_participant(embedding())[0], None)

    def test_server_opt_out_prevents_library_search(self):
        with patch.dict(os.environ, {"MEETING_AUTO_SPEAKER_MATCH": "0"}):
            processor = self.make_processor()
            processor.load_participants()
            self.assertFalse(processor._auto_profile_mode)

    def test_explicit_selection_is_not_auto_global(self):
        with patch.dict(os.environ, {"MEETING_AUTO_SPEAKER_MATCH": "1"}):
            processor = self.make_processor([self.person_id])
            processor.load_participants()
            self.assertFalse(processor._auto_profile_mode)
            self.assertEqual(processor.participant_ids, [self.person_id])
            self.assertEqual(processor._match_participant(embedding())[0], self.person_id)

    def test_manual_selection_skips_incompatible_embedding_model(self):
        with patch.dict(os.environ, {"MEETING_AUTO_SPEAKER_MATCH": "1"}):
            processor = self.make_processor([self.person_id])
            processor.load_participants()
            processor._participants[0]["vectors"].append({
                "emb": np.ones(128, dtype=np.float32),
                "embedding_model": "ecapa",
            })
            self.assertEqual(processor._match_participant(embedding())[0], self.person_id)

    def test_registered_name_survives_live_nemotron_projection(self):
        proc = self.make_processor()
        proc.settings = SimpleNamespace(sample_rate=16000)
        proc.segments = [{
            "id": 1, "start_sample": 0, "end_sample": 16000,
            "text": "test", "speaker_id": 2,
            "speaker_name": "Registered Person",
            "matched_profile_id": self.person_id,
            "matched_profile_name": "Registered Person",
            "speaker_match_similarity": 0.98,
            "cluster": -1, "translation": "", "language": "en",
        }]
        proc._speaker_map = {}
        proc._last_fingerprint = None
        proc._transcript_version = 0
        proc._independent_id_offset = 0
        proc._independent_pcm_origin_samples = 0
        proc._submit_shadow_samples = lambda: None
        sent = []
        proc._put = sent.append

        class AnonymousTrack:
            def activity_snapshot(self):
                return np.ones((16, 2), dtype=np.float32)
            def is_complete(self):
                return False

        proc.independent_diarization = AnonymousTrack()
        with patch("sessions.processor.speaker_evidence", return_value={1: {}}), \\
             patch("sessions.processor.speaker_stats",
                   return_value={"speaker_switches": 0, "unidentified_segments": 0}):
            self.assertTrue(proc._rebuild_and_push())

        self.assertEqual(len(sent), 1)
        item = sent[0]["segments"][0]
        self.assertEqual(item["speaker_name"], "Registered Person")
        self.assertEqual(item["speaker_id"], 2)
        self.assertEqual(item["speaker_diarization_source"], "registered_voiceprint")

    def test_all_touched_python_modules_are_syntax_valid(self):
        root = Path(__file__).resolve().parents[2]
        for filename in (
            "sessions/processor.py", "sessions/registered_identity.py",
            "storage/speakers.py", "storage/meetings.py",
            "api/meetings.py", "api/websocket.py", "sessions/session.py",
        ):
            with self.subTest(filename=filename):
                ast.parse((root / filename).read_text(encoding="utf8"), filename)


if __name__ == "__main__":
    unittest.main()

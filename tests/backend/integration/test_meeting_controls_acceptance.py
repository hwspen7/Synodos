"""End-to-end state contracts for edited speakers, saved settings and WAV replay."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.corrections import (
    RevisionConflict, assign_segment_speaker, correct_segment,
)
from storage.meetings import MeetingStore


def sample_state(mid: str) -> dict:
    return {
        "id": mid,
        "title": "Control acceptance",
        "sample_rate": 16000,
        "merge_gap_s": 2.0,
        "speaker_map": {},
        "participant_label_map": {},
        "segments": [{
            "id": 7, "start_sample": 0, "end_sample": 8000,
            "text": "Hello", "cluster": -11,
            "speaker_id": -1, "speaker_name": "Unidentified Speaker",
            "emb": None, "language": "en", "revision": 0,
        }],
    }


class ManualSpeakerAcceptanceTests(unittest.TestCase):
    def test_speaker_correction_persists_in_saved_and_exported_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MeetingStore(Path(temp))
            mid = store.create(title="Test")["id"]
            state = sample_state(mid)
            corrected = assign_segment_speaker(state, 7, "participant-1", "Alex", 0)
            self.assertEqual(corrected["speaker_name"], "Alex")
            self.assertEqual(corrected["speaker_id"], 1)
            self.assertEqual(corrected["revision"], 1)
            store.save_state(state)

            reopened = store.load_state(mid)
            self.assertTrue(reopened["segments"][0]["speaker_override"])
            self.assertEqual(reopened["segments"][0]["speaker_override_name"], "Alex")
            self.assertIn("Alex", store.export(mid, "md"))
            displayed = store.list_segments(mid)
            self.assertEqual(displayed[0]["speaker_name"], "Alex")
            self.assertEqual(displayed[0]["revision"], 1)
            with self.assertRaises(RevisionConflict):
                assign_segment_speaker(reopened, 7, None, None, 0)

    def test_manual_override_survives_resume_and_reclustering(self):
        settings = Settings()
        state = sample_state("manual-override")
        assign_segment_speaker(state, 7, "participant-1", "Alex", 0)
        proc = SessionProcessor(
            "test-speaker", AudioBuffer(settings.sample_rate, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        proc.load_state(state)
        proc._recluster()
        proc._rebuild_and_push()
        self.assertEqual(proc.segments[0]["speaker_name"], "Alex")
        self.assertEqual(proc.segments[0]["speaker_id"], 1)
        snapshot = proc.snapshot_state()
        self.assertTrue(snapshot["segments"][0]["speaker_override"])
        self.assertEqual(snapshot["segments"][0]["speaker_override_name"], "Alex")

    def test_manual_text_review_flag_survives_meeting_resume(self):
        settings = Settings()
        state = sample_state("review-flag")
        correct_segment(state, 7, "Hello there", 0)
        processor = SessionProcessor(
            "manual-text", AudioBuffer(settings.sample_rate, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        processor.load_state(state)
        snapshot = processor.snapshot_state()
        self.assertEqual(snapshot["segments"][0]["text"], "Hello there")
        self.assertEqual(snapshot["segments"][0]["revision"], 1)
        self.assertTrue(snapshot["segments"][0]["manually_corrected"])

    def test_text_and_speaker_edits_preserve_separate_revisions(self):
        state = sample_state("manual-revisions")
        assign_segment_speaker(state, 7, "participant-1", "Alex", 0)
        corrected = correct_segment(state, 7, "Hello there", 1)
        self.assertEqual(corrected["revision"], 2)
        self.assertEqual(corrected["speaker_override_name"], "Alex")
        assign_segment_speaker(state, 7, None, None, 2)
        self.assertEqual(corrected["speaker_name"], "Unidentified Speaker")
        self.assertEqual(corrected["revision"], 3)


class SavedConfigurationAcceptanceTests(unittest.TestCase):
    def test_settings_remain_after_meeting_state_snapshot(self):
        settings = Settings()
        state = sample_state("settings")
        state["ui_settings"] = {
            "language": "en", "translate_to": "zh",
            "record_audio": True, "vad_threshold": 4,
            "rec_threshold": 1.5,
        }
        proc = SessionProcessor(
            "test-settings", AudioBuffer(settings.sample_rate, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
        proc.load_state(state)
        self.assertEqual(proc.snapshot_state()["ui_settings"], state["ui_settings"])


class WavListingAcceptanceTests(unittest.TestCase):
    def test_recording_listing_returns_playable_audio_metadata(self):
        from api import speech_features

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MeetingStore(root / "meetings")
            mid = store.create(title="WAV test")["id"]
            folder = root / "recordings" / mid
            folder.mkdir(parents=True)
            filename = "a" * 32 + ".wav"
            with wave.open(str(folder / filename), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            class Config:
                recording_root = folder.parent

            with patch.object(speech_features, "meeting_store", store), \
                 patch.object(speech_features, "load_settings", return_value=Config()):
                result = asyncio.run(speech_features.list_recordings(mid))
            self.assertEqual(len(result["recordings"]), 1)
            entry = result["recordings"][0]
            self.assertEqual(entry["name"], filename)
            self.assertEqual(entry["duration_s"], 1.0)
            self.assertGreater(entry["bytes"], 32000)
            self.assertTrue(entry["modified_at"])




class OfflineMeetingSettingsTests(unittest.TestCase):
    def test_settings_api_does_not_erase_existing_transcript(self):
        from unittest.mock import patch
        from api import meetings as meeting_routes
        from storage.speakers import SpeakerStore
        from storage.hotwords import HotwordStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MeetingStore(root / "meetings")
            sid = store.create(title="Settings test")["id"]
            state = sample_state(sid)
            store.save_state(state)

            speakers = SpeakerStore(root / "speakers")
            enrolled = speakers.create("Alex")
            hotwords = HotwordStore(root / "hotwords")
            with patch.object(meeting_routes, "meeting_store", store), \
                 patch.object(meeting_routes, "speaker_store", speakers), \
                 patch.object(meeting_routes, "hotword_store", hotwords):
                payload = meeting_routes.MeetingSettingsPayload(
                    participant_ids=[enrolled["id"]],
                    language="en", translate_to="zh", record_audio=True,
                    vad_threshold=4.0, rec_threshold=1.5,
                )
                result = asyncio.run(meeting_routes.update_meeting_settings(sid, payload))

            self.assertTrue(result["ok"])
            self.assertEqual(store.get_meeting(sid)["ui_settings"]["translate_to"], "zh")
            self.assertEqual(store.get_meeting(sid)["participant_ids"], [enrolled["id"]])
            self.assertEqual(store.load_state(sid)["segments"][0]["text"], "Hello")


if __name__ == "__main__":
    unittest.main()

"""Saved meeting transcripts expose evidence without embedding matrices."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

from api.meetings import get_meeting_segments
from api.websocket import resolve_record_audio
from storage.meetings import MeetingStore


class MeetingEvidenceExportTests(unittest.TestCase):
    def test_saved_segments_and_json_export_include_uncertainty_and_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MeetingStore(folder)
            mid = store.create(title="Audio evidence")["id"]
            state = {
                "id": mid,
                "title": "Audio evidence",
                "sample_rate": 16000,
                "merge_gap_s": 2.0,
                "speaker_map": {2: 4},
                "speech_coverage_gaps": [{
                    "start_ms": 1000, "end_ms": 2400,
                    "status": "voice_detected_transcript_unavailable",
                }],
                "segments": [{
                    "id": 1, "start_sample": 0, "end_sample": 16000,
                    "text": "Recovered exact words", "cluster": 2,
                    "speaker_id": 4, "speaker_name": "Speaker 4",
                    "emb": [0.1, 0.2, 0.3, 0.4],
                    "recognition_origin": "gap_recovery",
                    "asr_confidence": None,
                    "speaker_status": "tentative",
                    "speaker_similarity": 0.67,
                    "speaker_margin": 0.05,
                    "speaker_evidence_count": 1,
                    "speaker_evidence_status": "acoustic_backfill",
                }],
            }
            store.save_state(state)
            listed = store.list_segments(mid)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["speaker_status"], "tentative")
            self.assertEqual(listed[0]["recognition_origin"], "gap_recovery")
            self.assertEqual(listed[0]["speaker_similarity"], 0.67)
            self.assertEqual(listed[0]["speaker_evidence_count"], 1)
            self.assertEqual(listed[0]["speaker_score_kind"],
                             "raw_cosine_not_probability")
            self.assertNotIn("emb", listed[0])
            self.assertEqual(
                store.list_speech_coverage_gaps(mid)[0]["start_ms"], 1000
            )
            exported = json.loads(store.export(mid, "json"))
            self.assertEqual(
                exported["transcript"]["segments"][0]["recognition_origin"],
                "gap_recovery",
            )
            self.assertEqual(
                exported["transcript"]["speech_coverage_gaps"],
                state["speech_coverage_gaps"],
            )
            self.assertNotIn('"emb":', store.export(mid, "json"))
            with patch("api.meetings.meeting_store", store):
                api_data = asyncio.run(get_meeting_segments(mid))
            self.assertEqual(api_data["speech_coverage_gaps"],
                             state["speech_coverage_gaps"])
            self.assertEqual(api_data["segments"][0]["speaker_status"],
                             "tentative")

    def test_recording_consent_uses_explicit_override_or_saved_boolean(self):
        saved = {"ui_settings": {"record_audio": True}}
        self.assertTrue(resolve_record_audio(None, saved, False))
        self.assertFalse(resolve_record_audio(False, saved, True))
        self.assertTrue(resolve_record_audio(True, None, False))
        self.assertFalse(resolve_record_audio(None, None, False))
        self.assertTrue(resolve_record_audio(None, None, True))
        self.assertFalse(resolve_record_audio(
            None, {"ui_settings": {"record_audio": "false"}}, False
        ))


if __name__ == "__main__":
    unittest.main()

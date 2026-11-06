"""Numbered caption assignments stay separate from acoustic evidence."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.provisional_speakers import (
    PROVISIONAL_EXISTING, PROVISIONAL_NEW, assign_provisional_speakers,
)
from speech.config import SpeechSettings
from storage.meetings import MeetingStore


def row(segment_id, start, end, speaker=-1, **extra):
    return {
        "id": segment_id, "start_ms": start, "end_ms": end, "speaker_id": speaker,
        "speaker_name": f"Speaker {speaker}" if speaker > 0 else "Unidentified Speaker",
        **extra,
    }


def channel(number, **extra):
    return {
        "speaker_acoustic_channel": number,
        "speaker_activity_status": "official_turn_primary",
        "speaker_activity_score": 0.91,
        "speaker_activity_margin": 0.57,
        **extra,
    }


def persisted(item):
    return {
        **item, "start_sample": item["start_ms"] * 16, "end_sample": item["end_ms"] * 16,
        "text": f"Turn {item['id']}", "cluster": -3, "emb": None,
    }


class ProvisionalSpeakerTests(unittest.TestCase):
    def test_channel_evidence_reuses_known_voice_and_groups_new_channel(self):
        items = [
            row(1, 0, 1000, 3, speaker_diarization_source="nemotron3", **channel(7)),
            row(2, 1800, 2100, **channel(7)),
            row(3, 3500, 4100, **channel(2)),
            row(4, 8000, 8800, **channel(2)),
        ]
        actual = assign_provisional_speakers(items, reserved_labels=[1])
        self.assertEqual(actual, {2: 3, 3: 2, 4: 2})
        self.assertEqual(items[1]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)
        self.assertEqual(items[2]["speaker_final_resolution_status"], PROVISIONAL_NEW)
        self.assertTrue(all(item["speaker_uncertain"] for item in items[1:]))

    def test_conflicts_and_overlap_cannot_inherit_existing_speaker(self):
        items = [
            row(1, 0, 1000, 3),
            row(2, 1050, 1900, **channel(7,
                speaker_channel_gate_reason="existing_channel_voice_conflict")),
            row(3, 1950, 3000, 3),
            row(4, 2900, 3600, overlap_intervals_ms=[[2900, 3000]]),
        ]
        assign_provisional_speakers(items, channel_labels={7: 3})
        self.assertEqual([item["speaker_id"] for item in items], [3, 1, 3, 2])

    def test_brief_bridge_uses_existing_label_but_long_gap_does_not(self):
        items = [
            row(1, 0, 1000, 3),
            row(2, 1200, 1700),
            row(3, 1900, 3000, 3),
            row(4, 4400, 6100),
            row(5, 7000, 8000, 3),
        ]
        assign_provisional_speakers(items)
        self.assertEqual([item["speaker_id"] for item in items], [3, 3, 3, 1, 3])
        self.assertEqual(items[1]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)

    def test_contiguous_fragments_reuse_their_label_when_a_new_turn_arrives(self):
        items = [row(2, 1000, 1800), row(3, 1900, 2400)]
        first = assign_provisional_speakers(items, reserved_labels=[1])
        later = [row(1, 0, 500), row(2, 1000, 1800), row(3, 1900, 2400)]
        self.assertEqual(first, {2: 2, 3: 2})
        self.assertEqual(assign_provisional_speakers(later, previous=first, reserved_labels=[1]),
                         {1: 3, 2: 2, 3: 2})

    def test_voice_match_requires_a_clear_lead_over_other_people(self):
        items = [
            row(1, 0, 1500, 3, speaker_evidence_status="turn_voiceprint"),
            row(2, 2000, 3400, 5, speaker_evidence_status="turn_voiceprint"),
            row(3, 4100, 5000),
            row(4, 6000, 7000),
        ]
        actual = assign_provisional_speakers(items, voice_embeddings={
            1: [1, 0], 2: [0, 1], 3: [.98, .20], 4: [1, 1],
        })
        self.assertEqual(actual, {3: 3, 4: 1})
        self.assertEqual(items[2]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)
        self.assertEqual(items[3]["speaker_final_resolution_status"], PROVISIONAL_NEW)

    def test_clearing_enrolled_identity_returns_a_numbered_caption(self):
        from api.speech_features import SpeakerCorrectionRequest, edit_segment_speaker

        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(Path(root))
            store.save_state({"id": "clear-identity", "segments": [persisted(row(1, 0, 1000, 4))]})
            with patch("api.speech_features.meeting_store", store):
                result = asyncio.run(edit_segment_speaker(
                    "clear-identity", 1, SpeakerCorrectionRequest(revision=0),
                ))
            self.assertGreater(result["speaker_id"], 0)
            self.assertEqual(result["speaker_final_resolution_status"], PROVISIONAL_NEW)
            self.assertEqual(result["revision"], 1)

    def test_saved_reads_and_exports_number_legacy_records_without_rewriting_them(self):
        items = [persisted(row(1, 0, 1000, 3)), persisted(row(2, 2500, 3100))]
        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(Path(root))
            store.save_state({"id": "legacy-caption", "segments": items})
            shown = store.list_segments("legacy-caption")
            exported = json.loads(store.export("legacy-caption", fmt="json"))
            self.assertEqual([item["speaker_id"] for item in shown], [3, 1])
            self.assertEqual(exported["transcript"]["segments"], shown)
            self.assertEqual(store.load_state("legacy-caption")["segments"][1]["speaker_id"], -1)

    def test_resume_preserves_display_id_without_seeding_voice_clustering(self):
        settings = Settings()
        with patch("sessions.processor.load_settings", return_value=SpeechSettings()):
            processor = SessionProcessor("resume", AudioBuffer(16000), settings,
                                         object(), asyncio.Queue())
        item = persisted(row(1, 0, 1000, 8, speaker_final_resolution_status=PROVISIONAL_NEW,
                             speaker_uncertain=True))
        processor.load_state({"segments": [item]})
        self.assertEqual(processor.segments[0]["speaker_id"], -1)
        self.assertNotIn(8, processor._speaker_birth_approved)
        processor._rebuild_and_push()
        shown = processor.snapshot_state()["segments"][0]
        self.assertEqual((shown["speaker_id"], shown["speaker_name"]), (8, "Speaker 8"))
        self.assertEqual(shown["speaker_final_resolution_status"], PROVISIONAL_NEW)

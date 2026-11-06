"""Numbered caption assignments stay separate from acoustic evidence."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.channels.provisional_speakers import (
    ATTRIBUTION_DEFERRED, PROVISIONAL_EXISTING, PROVISIONAL_NEW,
    assign_provisional_speakers,
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
    def test_channel_evidence_reuses_known_voice_and_defers_unbound_channel(self):
        items = [
            row(1, 0, 1000, 3, speaker_diarization_source="nemotron3", **channel(7)),
            row(2, 1800, 2100, **channel(7)),
            row(3, 3500, 4100, **channel(2)),
            row(4, 8000, 8800, **channel(2)),
        ]
        actual = assign_provisional_speakers(items)
        self.assertEqual(actual, {2: 3})
        self.assertEqual(items[1]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)
        self.assertEqual(items[2]["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
        self.assertTrue(all(item["speaker_uncertain"] for item in items[1:]))

    def test_raw_channel_does_not_cross_recording_streams(self):
        items = [
            row(1, 0, 1000, 1, speaker_diarization_source="nemotron3", **channel(1)),
            row(2, 3000, 4000, **channel(1, speaker_acoustic_stream_start=32000)),
        ]
        self.assertEqual(assign_provisional_speakers(items), {})
        self.assertEqual(items[1]["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
        self.assertEqual(assign_provisional_speakers(
            items, channel_labels={1: 2}, channel_stream_start=32000), {2: 2})

    def test_conflicts_and_overlap_cannot_inherit_existing_speaker(self):
        items = [
            row(1, 0, 1000, 3),
            row(2, 1050, 1900, **channel(7,
                speaker_channel_gate_reason="existing_channel_voice_conflict")),
            row(3, 1950, 3000, 3),
            row(4, 2900, 3600, overlap_intervals_ms=[[2900, 3000]]),
        ]
        assign_provisional_speakers(items, channel_labels={7: 3})
        self.assertEqual([item["speaker_id"] for item in items], [3, -1, 3, -1])

    def test_brief_bridge_uses_existing_label_but_long_gap_does_not(self):
        items = [
            row(1, 0, 1000, 3),
            row(2, 1200, 1700),
            row(3, 1900, 3000, 3),
            row(4, 4400, 6100),
            row(5, 7000, 8000, 3),
        ]
        assign_provisional_speakers(items)
        self.assertEqual([item["speaker_id"] for item in items], [3, 3, 3, -1, 3])
        self.assertEqual(items[1]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)

    def test_later_speakers_cannot_renumber_an_unsupported_turn(self):
        items = [row(1, 0, 1000, 1), row(2, 1500, 3000)]
        self.assertEqual(assign_provisional_speakers(items), {})
        items[1].update(speaker_id=2, speaker_final_resolution_status=PROVISIONAL_NEW)
        items.extend(row(i + 1, i * 2000, i * 2000 + 1000, i) for i in (2, 3, 4))
        self.assertEqual(assign_provisional_speakers(items), {})
        self.assertEqual(items[1]["speaker_id"], -1)
        self.assertEqual(items[1]["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
        self.assertEqual([item["speaker_id"] for item in items[2:]], [2, 3, 4])

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
        self.assertEqual(actual, {3: 3})
        self.assertEqual(items[2]["speaker_final_resolution_status"], PROVISIONAL_EXISTING)
        self.assertEqual(items[3]["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)

    def test_unsupported_short_turn_does_not_allocate_a_new_speaker(self):
        items = [
            row(27, 72000, 73000, 2),
            row(29, 76000, 77000, speaker_evidence_status="short_turn",
                language="zh", language_confidence=.99),
            row(28, 77000, 80000, 2),
        ]
        self.assertEqual(assign_provisional_speakers(items), {})
        self.assertEqual(items[1]["speaker_id"], -1)
        self.assertEqual(items[1]["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
        items[1].update(channel(2))
        self.assertEqual(assign_provisional_speakers(items, channel_labels={2: 2}), {29: 2})

    def test_confident_language_can_corroborate_only_the_acoustic_winner(self):
        def reference(i, start, label, language):
            return row(i, start, start + 1300, label,
                       speaker_evidence_status="turn_voiceprint",
                       language=language, language_confidence=.97)

        def assign(probe, language, confidence):
            items = [
                reference(1, 0, 2, "zh"),
                reference(2, 1500, 2, "zh"),
                reference(3, 3000, 3, "en"),
                reference(4, 4500, 3, "en"),
                row(5, 6200, 7000, language=language, language_confidence=confidence),
            ]
            assign_provisional_speakers(items, voice_embeddings={
                1: [1, 0], 2: [1, 0], 3: [.995, .10], 4: [.995, .10], 5: probe,
            })
            return items[-1]["speaker_id"]

        self.assertEqual(assign([.8, -.6], "zh", .99), 2)
        self.assertEqual(assign([.8, -.6], "en", .99), -1)
        self.assertEqual(assign([.8, -.6], "zh", .70), -1)
        self.assertEqual(assign([.6, -.8], "zh", .99), -1)

    def test_saved_unsupported_placeholder_is_deferred_without_erasing_text(self):
        item = persisted(row(1, 0, 800, 5, speaker_evidence_status="short_turn",
                             speaker_final_resolution_status=PROVISIONAL_NEW))
        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(Path(root))
            store.save_state({"id": "short-turn", "segments": [item]})
            shown = store.list_segments("short-turn")[0]
            self.assertEqual(shown["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
            self.assertEqual(shown["text"], item["text"])
            self.assertEqual(store.load_state("short-turn")["segments"][0]["speaker_id"], 5)

    def test_clearing_enrolled_identity_defers_an_unsupported_caption(self):
        from api.speech_features import SpeakerCorrectionRequest, edit_segment_speaker

        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(Path(root))
            store.save_state({"id": "clear-identity", "segments": [persisted(row(1, 0, 1000, 4))]})
            with patch("api.speech_features.meeting_store", store):
                result = asyncio.run(edit_segment_speaker(
                    "clear-identity", 1, SpeakerCorrectionRequest(revision=0),
                ))
            self.assertEqual(result["speaker_id"], -1)
            self.assertEqual(result["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
            self.assertEqual(result["revision"], 1)

    def test_saved_reads_and_exports_keep_unassigned_content_without_rewriting_it(self):
        items = [persisted(row(1, 0, 1000, 3)), persisted(row(2, 2500, 3100))]
        with tempfile.TemporaryDirectory() as root:
            store = MeetingStore(Path(root))
            store.save_state({"id": "legacy-caption", "segments": items})
            shown = store.list_segments("legacy-caption")
            exported = json.loads(store.export("legacy-caption", fmt="json"))
            self.assertEqual([item["speaker_id"] for item in shown], [3, -1])
            self.assertEqual(exported["transcript"]["segments"], shown)
            self.assertEqual(store.load_state("legacy-caption")["segments"][1]["speaker_id"], -1)

    def test_resume_does_not_restore_unsupported_short_speaker(self):
        with patch("sessions.processor.load_settings", return_value=SpeechSettings()):
            processor = SessionProcessor("resume", AudioBuffer(16000), Settings(),
                                         object(), asyncio.Queue())
        item = persisted(row(29, 76000, 77000, 5, speaker_evidence_status="short_turn",
                             speaker_final_resolution_status=PROVISIONAL_NEW,
                             speaker_uncertain=True))
        processor.load_state({"segments": [item]})
        events = []
        processor._put = events.append
        processor._rebuild_and_push()
        live = next(event["segments"][0] for event in events if event["type"] == "transcript")
        for segment in (live, processor.snapshot_state()["segments"][0]):
            self.assertEqual(segment["speaker_id"], -1)
            self.assertEqual(segment["speaker_final_resolution_status"], ATTRIBUTION_DEFERRED)
            self.assertEqual(segment["text"], item["text"])

    def test_resume_preserves_existing_match_without_seeding_voice_clustering(self):
        settings = Settings()
        with patch("sessions.processor.load_settings", return_value=SpeechSettings()):
            processor = SessionProcessor("resume", AudioBuffer(16000), settings,
                                         object(), asyncio.Queue())
        item = persisted(row(1, 0, 1000, 8, speaker_final_resolution_status=PROVISIONAL_EXISTING,
                             speaker_uncertain=True))
        processor.load_state({"segments": [item]})
        self.assertEqual(processor.segments[0]["speaker_id"], -1)
        self.assertNotIn(8, processor._speaker_birth_approved)
        processor._rebuild_and_push()
        shown = processor.snapshot_state()["segments"][0]
        self.assertEqual((shown["speaker_id"], shown["speaker_name"]), (8, "Speaker 8"))
        self.assertEqual(shown["speaker_final_resolution_status"], PROVISIONAL_EXISTING)

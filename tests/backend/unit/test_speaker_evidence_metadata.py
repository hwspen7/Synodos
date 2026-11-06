"""Raw acoustic speaker evidence, not calibrated verification probabilities."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from sessions.audio_buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from sessions.speaker_evidence import speaker_evidence
from speech.config import SpeechSettings


def vec(index, *, dim=64):
    values = np.zeros(dim, dtype=np.float32)
    values[index] = 1.0
    return values


class SpeakerEvidenceTests(unittest.TestCase):
    def test_supported_same_speaker_has_raw_cosine_not_probability(self):
        records = [
            {"id": 1, "speaker_id": 1, "cluster": 0, "emb": vec(0)},
            {"id": 2, "speaker_id": 1, "cluster": 0, "emb": vec(0)},
            {"id": 3, "speaker_id": 2, "cluster": 1, "emb": vec(1)},
            {"id": 4, "speaker_id": 2, "cluster": 1, "emb": vec(1)},
        ]
        evidence = speaker_evidence(records)
        self.assertEqual(evidence[1]["speaker_status"], "supported")
        self.assertEqual(evidence[1]["speaker_similarity"], 1.0)
        self.assertEqual(evidence[1]["speaker_margin"], 1.0)
        self.assertEqual(evidence[1]["speaker_evidence_count"], 1)
        self.assertEqual(evidence[1]["speaker_score_kind"],
                         "raw_cosine_not_probability")

    def test_unknown_tentative_and_enrolled_speech_are_distinguished(self):
        records = [
            {"id": 1, "speaker_id": -1, "cluster": -3, "emb": None,
             "speaker_uncertain": True},
            {"id": 2, "speaker_id": 1, "cluster": 0, "emb": vec(0)},
            {"id": 3, "speaker_id": 7, "cluster": -1, "emb": vec(1),
             "speaker_match_similarity": 0.73},
            {"id": 4, "speaker_id": 1, "cluster": 0, "emb": vec(1),
             "speaker_uncertain": True},
        ]
        result = speaker_evidence(records)
        self.assertEqual(result[1]["speaker_status"], "unidentified")
        self.assertIsNone(result[1]["speaker_similarity"])
        self.assertEqual(result[2]["speaker_status"], "tentative")
        self.assertEqual(result[3]["speaker_status"], "enrolled_match")
        self.assertEqual(result[3]["speaker_similarity"], 0.73)
        self.assertEqual(result[4]["speaker_status"], "uncertain")

    def test_mixed_voiceprint_model_sizes_cannot_break_transcript(self):
        records = [
            {"id": 1, "speaker_id": 1, "cluster": 0, "emb": vec(0, dim=64)},
            {"id": 2, "speaker_id": 1, "cluster": 0, "emb": vec(0, dim=64)},
            {"id": 3, "speaker_id": 1, "cluster": 0, "emb": vec(0, dim=128)},
            {"id": 4, "speaker_id": 2, "cluster": 1, "emb": vec(1, dim=128)},
        ]
        output = speaker_evidence(records)
        self.assertEqual(output[1]["speaker_similarity"], 1.0)
        self.assertEqual(output[3]["speaker_status"], "tentative")
        self.assertEqual(output[3]["speaker_evidence_count"], 0)
        self.assertEqual(output[4]["speaker_status"], "tentative")

    def test_transcript_exposes_raw_similarity_and_asr_provenance(self):
        settings = Settings()
        with patch("sessions.processor.load_settings",
                   return_value=SpeechSettings(engine="legacy")):
            processor = SessionProcessor(
                "metadata", AudioBuffer(settings.sample_rate), settings,
                object(), asyncio.Queue()
            )
        events = []
        processor._put = events.append
        processor._submit_shadow_samples = lambda: None
        processor.segments = [
            {
                "id": 1, "cluster": 0, "speaker_id": 1,
                "speaker_name": "Speaker 1",
                "start_sample": 0, "end_sample": 16000,
                "text": "Confirmed words", "language": "en",
                "emb": vec(0),
                "speaker_evidence_status": "turn_voiceprint",
                "asr_confidence": None,
                "recognition_origin": "gap_recovery",
            },
            {
                "id": 2, "cluster": 0, "speaker_id": 1,
                "speaker_name": "Speaker 1",
                "start_sample": 16000, "end_sample": 32000,
                "text": "More words", "language": "en",
                "emb": vec(0),
                "speaker_evidence_status": "turn_voiceprint",
                "recognition_origin": "primary",
            },
        ]
        self.assertTrue(processor._rebuild_and_push())
        transcript = next(e for e in events if e["type"] == "transcript")
        first = transcript["segments"][0]
        self.assertEqual(first["recognition_origin"], "gap_recovery")
        self.assertEqual(first["speaker_status"], "supported")
        self.assertEqual(first["speaker_similarity"], 1.0)
        self.assertIsNone(first["text_confidence"])
        self.assertFalse(processor._rebuild_and_push())
        saved = processor.snapshot_state()
        self.assertEqual(saved["segments"][0]["recognition_origin"], "gap_recovery")
        self.assertEqual(saved["segments"][0]["speaker_evidence_status"],
                         "turn_voiceprint")


if __name__ == "__main__":
    unittest.main()

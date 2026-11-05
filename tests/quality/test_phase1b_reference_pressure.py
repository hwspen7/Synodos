"""No-GPU regression tests for AMI reference overlap diagnostics."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import unittest

from tools.ami_shadow10_prepare import SELECTED
from tools.phase1b_reference_pressure import diagnose, reference_pressure


def trial(meeting: str) -> dict:
    return {
        "schema": "meeting-phase1b-frozen-diarization-v1",
        "model": "pyannote",
        "meeting": meeting,
        "source_audio_sha256": "a" * 64,
        "reference_source_sha256": "b" * 64,
        "frozen_case_sha256": "c" * 64,
        "frame_step_ms": 100,
        "collar_ms": 0,
        "reference_protocol": "only_words",
        "overlap_included": True,
        "metrics": {
            "der": 0.2,
            "missed_speaker_time": 0.1,
            "false_alarm_time": 0.06,
            "confusion_time": 0.04,
            "reference_speaker_seconds": 2.0,
        },
        "diagnostics": {
            "model_id": "pyannote/speaker-diarization-community-1",
            "device": "cuda",
            "predicted_speakers": 2,
        },
        "accepted_acoustic_candidate": False,
    }


class ReferencePressureTests(unittest.TestCase):
    def test_overlap_is_counted_as_extra_reference_speaker_time(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
        ]
        measured = reference_pressure(turns)
        self.assertEqual(measured["reference_speaker_seconds"], 2.0)
        self.assertEqual(measured["reference_overlap_wall_seconds"], 0.5)
        self.assertEqual(measured["reference_overlap_extra_speaker_seconds"], 0.5)
        self.assertEqual(measured["reference_extra_speaker_fraction"], .25)
        self.assertEqual(measured["reference_overlap_fraction_of_speech_wall"], .33333)

    def test_duplicate_reference_speaker_does_not_count_as_overlap(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 500, "end_ms": 1500, "speaker": "A"},
        ]
        measured = reference_pressure(turns)
        self.assertEqual(measured["reference_speaker_seconds"], 1.5)
        self.assertEqual(measured["reference_overlap_wall_seconds"], 0)
        self.assertEqual(measured["reference_speaker_count"], 1)

    def test_join_real_report_protocol_and_compute_en2002_group(self):
        meetings = [meeting for meeting, _ in SELECTED]
        cases = {meeting: {
            "reference_turns": [
                {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
                {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
            ]
        } for meeting in meetings}
        with patch("tools.phase1b_pyannote_ami10.pinned_case", return_value=(
            Path("fake.wav"),
            {"sha256": "a" * 64, "reference_source_sha256": "b" * 64},
            "c" * 64,
        )):
            result = diagnose([trial(meeting) for meeting in meetings],
                              cases, Path("unused"))
        self.assertEqual(result["meeting_count"], 10)
        en = next(x for x in result["per_source_group"]
                  if x["source_group"] == "EN2002")
        self.assertEqual(en["meeting_count"], 3)
        self.assertEqual(en["weighted_metrics"]["der"], .2)
        self.assertEqual(result["per_meeting"][0]["reference_extra_speaker_fraction"], .25)

    def test_reject_corrupt_reference_or_count_disagreement(self):
        cases = {
            meeting: {"reference_turns": [
                {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
                {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
            ]} for meeting, _ in SELECTED
        }
        cases["EN2002a"]["reference_turns"][0]["end_ms"] = 900
        with patch("tools.phase1b_pyannote_ami10.pinned_case", return_value=(
            Path("fake.wav"),
            {"sha256": "a" * 64, "reference_source_sha256": "b" * 64},
            "c" * 64,
        )):
            with self.assertRaisesRegex(ValueError, "reference speaker time"):
                diagnose([trial(m) for m, _ in SELECTED],
                         cases, Path("unused"))


if __name__ == "__main__":
    unittest.main()

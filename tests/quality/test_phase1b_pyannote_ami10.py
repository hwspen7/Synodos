"""CPU-only tests for the real-model AMI10 cohort orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.ami_shadow10_prepare import SELECTED
from tools.phase1b_pyannote_ami10 import (
    execute, summarize, validate_report,
)


def stub_pinned_case(_root, _meeting):
    return (
        Path("unused.wav"),
        {
            "sha256": "a" * 64,
            "reference_source_sha256": "b" * 64,
        },
        "c" * 64,
    )


def report(meeting, *, der=0.13, seconds=100.0):
    return {
        "schema": "meeting-phase1b-frozen-diarization-v1",
        "meeting": meeting,
        "model": "pyannote",
        "source_audio_sha256": "a" * 64,
        "reference_source_sha256": "b" * 64,
        "frozen_case_sha256": "c" * 64,
        "frame_step_ms": 100,
        "collar_ms": 0,
        "reference_protocol": "only_words",
        "overlap_included": True,
        "metrics": {
            "der": der,
            "missed_speaker_time": der / 2,
            "false_alarm_time": der / 4,
            "confusion_time": der / 4,
            "reference_speaker_seconds": seconds,
        },
        "diagnostics": {
            "device": "cuda",
            "model_id": "pyannote/speaker-diarization-community-1",
            "wall_seconds_including_load": 3.0,
        },
        "accepted_acoustic_candidate": der < 0.20,
    }


class FrozenPyannoteCohortTests(unittest.TestCase):
    def test_weighted_der_and_single_meeting_status_are_separate(self):
        rows = [
            report(m, der=.30 if m == "TS3003b" else .10,
                   seconds=300 if m == "TS3003b" else 100)
            for m, _ in SELECTED
        ]
        with patch("tools.phase1b_pyannote_ami10.pinned_case",
                   side_effect=stub_pinned_case):
            result = summarize(rows, Path("unused"))
        self.assertEqual(result["meeting_count"], 10)
        self.assertEqual(result["weighted_metrics"]["der"], .15)
        self.assertTrue(result["overall_acoustic_goal_met"])
        self.assertFalse(result["all_meetings_below_20_percent_der"])
        self.assertEqual(result["meetings_below_20_percent_der"], 9)
        self.assertFalse(result["automatic_model_promotion"])

    def test_cannot_approve_missing_duplicate_or_tampered_evidence(self):
        names = [m for m, _ in SELECTED]
        rows = [report(m) for m in names]
        with patch("tools.phase1b_pyannote_ami10.pinned_case",
                   side_effect=stub_pinned_case):
            for bad in (rows[:-1], rows[:-1] + [rows[0]]):
                with self.assertRaises(ValueError):
                    summarize(bad, Path("unused"))
            bad = report(names[0])
            bad["source_audio_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "source_audio_sha256"):
                validate_report(bad, Path("unused"), names[0])
            bad = report(names[0])
            bad["diagnostics"]["device"] = "cpu"
            with self.assertRaisesRegex(ValueError, "not from CUDA"):
                validate_report(bad, Path("unused"), names[0])
            bad = report(names[0])
            bad["metrics"]["der"] = .01
            with self.assertRaisesRegex(ValueError, "component totals"):
                validate_report(bad, Path("unused"), names[0])

    def test_complete_resume_uses_only_real_saved_reports_without_gpu(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ami_shadow10"
            run = root / "phase1b" / "pyannote10-test"
            run.mkdir(parents=True)
            for meeting, _ in SELECTED:
                (run / f"pyannote-{meeting}.json").write_text(
                    json.dumps(report(meeting)), encoding="utf-8",
                )
            with patch("tools.phase1b_pyannote_ami10.pinned_case",
                       side_effect=stub_pinned_case), patch(
                "tools.phase1b_pyannote_ami10._inside",
                return_value=True,
            ):
                summary = execute(root, run, resume=True)
            self.assertTrue(summary["overall_acoustic_goal_met"])
            self.assertTrue((run / "pyannote_ami10_summary.json").is_file())
            self.assertEqual(summary["meetings_below_20_percent_der"], 10)
            with patch("tools.phase1b_pyannote_ami10.pinned_case",
                       side_effect=stub_pinned_case), patch(
                "tools.phase1b_pyannote_ami10._inside", return_value=True,
            ):
                with self.assertRaises(FileExistsError):
                    execute(root, run, resume=True)


if __name__ == "__main__":
    unittest.main()

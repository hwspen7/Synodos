"""Phase 1B diarization candidate tests without GPU or external weights."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from speech.phase1b_diarization_models import (
    _normalize_turns, _validate_model_path, ModelSetupError, run_model,
)
from tools.phase1b_compare import compare_reports
from tools.phase1b_diarization_benchmark import report_from_turns


@dataclass
class Span:
    start: float
    end: float


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        assert yield_label
        yield Span(0.0, 0.9), None, "voice_B"
        yield Span(0.6, 1.2), None, "voice_A"
        yield Span(1.3, 2.0), None, "voice_B"


def fake_report(meeting="TS3003b", model="nemotron3", *, der=0.15,
                audio="a"*64, reference="b"*64, case="c"*64):
    return {
        "schema": "meeting-phase1b-frozen-diarization-v1",
        "meeting": meeting, "model": model,
        "source_audio_sha256": audio,
        "reference_source_sha256": reference,
        "frozen_case_sha256": case,
        "frame_step_ms": 100, "collar_ms": 0,
        "reference_protocol": "only_words",
        "overlap_included": True,
        "metrics": {
            "der": der,
            "missed_speaker_time": 0.1,
            "false_alarm_time": 0.03,
            "confusion_time": 0.02,
            "reference_speaker_seconds": 200.0,
        }
    }


class Phase1BModelTests(unittest.TestCase):
    def test_anonymous_turns_preserve_overlap_and_stable_local_ids(self):
        result = _normalize_turns(FakeAnnotation(), 2000)
        self.assertEqual(len(result), 3)
        self.assertEqual([row["speaker_id"] for row in result], [1, 2, 1])
        self.assertLess(result[1]["start_ms"], result[0]["end_ms"])

    def test_bad_time_boundaries_rejected(self):
        for span in (Span(0.0, 5.0), Span(0.9, 0.1),
                     Span(float("nan"), 0.4)):
            class Bad:
                def itertracks(self, yield_label=False):
                    yield span, None, "A"
            with self.subTest(span=span), self.assertRaises(ValueError):
                _normalize_turns(Bad(), 2000)

    def test_no_implicit_model_download(self):
        with self.assertRaises(ModelSetupError):
            _validate_model_path(
                "pyannote/speaker-diarization-community-1", False
            )
        with self.assertRaises(ModelSetupError):
            _validate_model_path("unknown/repository", True)

    def test_injected_pipeline_does_not_download_models(self):
        with tempfile.TemporaryDirectory() as root:
            wav = Path(root) / "fake.wav"
            wav.write_bytes(b"not decoded by fake pipeline")
            turns, meta = run_model(
                "diarizen", wav, duration_ms=2000,
                pipeline=lambda _: FakeAnnotation()
            )
            self.assertEqual(meta["predicted_speakers"], 2)
            self.assertEqual(len(turns), 3)
            self.assertEqual(meta["inference_mode"], "offline_full_recording")

    def test_report_scoring_keeps_same_reference_protocol(self):
        case = {
            "source_meeting": "TS3003b",
            "sha256": "a" * 64,
            "reference_source_sha256": "b" * 64,
            "reference_turns": [
                {"speaker": "A", "start_ms": 0, "end_ms": 1000},
                {"speaker": "B", "start_ms": 500, "end_ms": 1500},
            ],
        }
        turns = [
            {"speaker_id": 1, "start_ms": 0, "end_ms": 1000},
            {"speaker_id": 2, "start_ms": 500, "end_ms": 1500},
        ]
        report = report_from_turns(case, "c"*64, "pyannote", turns, {})
        self.assertEqual(report["metrics"]["der"], 0)
        self.assertTrue(report["accepted_acoustic_candidate"])
        self.assertTrue(report["overlap_included"])


class Phase1BCompareTests(unittest.TestCase):
    def test_requires_complete_paired_models(self):
        rows = [
            fake_report(model=name)
            for name in ("nemotron3", "pyannote", "diarizen")
        ]
        result = compare_reports(rows, min_meetings=1)
        self.assertEqual(result["paired_meeting_count"], 1)
        self.assertTrue(result["acoustic_goal_met"])
        self.assertFalse(result["may_auto_promote_to_live"])
        self.assertFalse(compare_reports(rows)["complete_heldout_cohort"])

    def test_rejects_unpaired_reference_and_duplicate_model(self):
        rows = [
            fake_report(model=name)
            for name in ("nemotron3", "pyannote", "diarizen")
        ]
        with self.assertRaisesRegex(ValueError, "Nonpaired"):
            compare_reports(
                [rows[0], fake_report(model="pyannote", reference="d"*64),
                 rows[2]], min_meetings=1,
            )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_reports(rows + [rows[0]], min_meetings=1)

    def test_invalid_metric_or_protocol_never_approves(self):
        rows = [fake_report(model=m) for m in (
            "nemotron3", "pyannote", "diarizen"
        )]
        rows[1]["metrics"]["der"] = float("nan")
        with self.assertRaises(ValueError):
            compare_reports(rows, min_meetings=1)
        rows[1]["metrics"]["der"] = 0.14
        rows[1]["collar_ms"] = 250
        with self.assertRaisesRegex(ValueError, "protocol"):
            compare_reports(rows, min_meetings=1)


if __name__ == "__main__":
    unittest.main()

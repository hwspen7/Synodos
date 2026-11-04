"""Deterministic ASR gap-stage tracing without raw transcript storage."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speech.asr_stage_trace import (
    merged_intervals, model_intervals, utterance_intervals,
)
from speech.contracts import Utterance
from speech.multilingual_windows import recognize_windows
from tools.ami_asr_stage_gap_audit import audit_stages


class StageTraceTests(unittest.TestCase):
    def test_interval_union_and_text_free_model_timestamps(self):
        self.assertEqual(
            merged_intervals([(0, 500), (250, 1000), (1200, 1500)],
                             duration_ms=2000),
            [[0, 1000], [1200, 1500]],
        )
        pieces = [SimpleNamespace(start=0.2, end=0.8, text="SECRET")]
        self.assertEqual(model_intervals(pieces, duration_ms=2000),
                         [[200, 800]])
        self.assertEqual(utterance_intervals(
            [Utterance("SECRET", 200, 800)], duration_ms=2000),
            [[200, 800]],
        )

    def test_multilingual_window_records_raw_and_verified_time(self):
        def fake_transcribe(audio, **kwargs):
            return (
                iter([SimpleNamespace(
                    start=0.1, end=0.9, text="PRIVATE",
                    words=None,
                )]),
                SimpleNamespace(language="en", language_probability=0.99),
            )
        with patch.dict(os.environ, {
            "MEETING_ASR_TRACE_STAGES": "1",
        }):
            result = recognize_windows(
                np.ones(32000, dtype=np.float32),
                [{"start": 0, "end": 32000}],
                fake_transcribe, {},
            )
        trace = result.recovery_diagnostics["stage_trace"]
        self.assertTrue(trace["raw_model_intervals_ms"])
        self.assertTrue(trace["verified_intervals_ms"])
        self.assertNotIn("PRIVATE", json.dumps(trace))
        self.assertEqual(trace["speech_windows"], 1)

    def test_gap_audit_attributes_every_absent_reference_clock_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cases").mkdir()
            (root / "cases" / "TS3003b.json").write_text(
                json.dumps({"reference_turns": [
                    {"start_ms": 0, "end_ms": 2000, "speaker": "A"},
                ]}), encoding="utf-8",
            )
            replay = {
                "source_audio_duration_s": 240.0,
                "final_cluster_diagnostics": {
                    "anonymous_timeline": {
                        "segments": [{"start_ms": 0, "end_ms": 1000}],
                    },
                },
                "asr_coverage_trace": {
                    "schema": "meeting-asr-stage-trace-v1",
                    "truncated": False,
                    "cycles": [{
                        "vad_intervals_ms": [[0, 2000]],
                        "raw_model_intervals_ms": [[0, 1600]],
                        "verified_intervals_ms": [[0, 1400]],
                        "after_recovery_intervals_ms": [[0, 1400]],
                    }],
                },
            }
            path = root / "test_replay.json"
            path.write_text(json.dumps(replay), encoding="utf-8")
            mock_audit = {
                "source_audio_sha256": "a",
                "reference_source_sha256": "b",
                "reference_speech_without_captions": {
                    "reference_speech_without_captions_seconds": 1.0,
                },
            }
            with patch("tools.ami_asr_stage_gap_audit.score_case",
                       return_value=mock_audit):
                report = audit_stages(root, "TS3003b", path)
            self.assertEqual(report["reference_clock_no_caption_seconds"], 1.0)
            self.assertAlmostEqual(
                report["gap_causes_seconds"][
                    "vad_without_raw_model_timestamp"], 0.4,
            )
            self.assertAlmostEqual(
                report["gap_causes_seconds"][
                    "raw_model_without_verified_caption"], 0.2,
            )
            self.assertAlmostEqual(
                report["gap_causes_seconds"][
                    "verified_but_not_in_final_caption"], 0.4,
            )


if __name__ == "__main__":
    unittest.main()

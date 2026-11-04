"""Source-independent AMI scorer and fail-closed acceptance contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ami_nemotron_live_score import (
    _solo_only, _validated_independent, score_independent,
)


def complete_replay():
    return {
        "source_audio_duration_s": 240.0,
        "sent_samples": 3840000,
        "server_ready": {"independent_diarization": "nemotron3"},
        "independent_diarization": {
            "backend": "nemotron3",
            "complete": True, "state": "completed",
            "cross_chunk_identity_cache": True,
            "frame_step_ms": 10, "processed_audio_samples": 3840000,
            "scored_frames": 24000,
            "turns": [
                {"start_ms": 0, "end_ms": 120000, "speaker_id": 1},
                {"start_ms": 120000, "end_ms": 240000, "speaker_id": 2},
            ],
            "error": None,
        },
    }


class NemotronBenchmarkTests(unittest.TestCase):
    def test_complete_full_session_is_required(self):
        document = complete_replay()
        self.assertEqual(len(_validated_independent(document)), 2)
        document["independent_diarization"]["processed_audio_samples"] -= 3200
        with self.assertRaisesRegex(ValueError, "did not consume"):
            _validated_independent(document)

    def test_speaker_cache_and_model_source_required(self):
        document = complete_replay()
        document["independent_diarization"]["cross_chunk_identity_cache"] = False
        with self.assertRaisesRegex(ValueError, "cache continuity"):
            _validated_independent(document)

    def test_independent_metric_can_score_without_caption_diarization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cases").mkdir()
            reference = [
                {"start_ms": 0, "end_ms": 120000, "speaker": "A"},
                {"start_ms": 120000, "end_ms": 240000, "speaker": "B"},
            ]
            (root / "cases" / "TS3003b.json").write_text(
                json.dumps({"reference_turns": reference}), encoding="utf-8",
            )
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(complete_replay()),
                                   encoding="utf-8")
            mock_audit = {
                "source_audio_sha256": "sha-audio",
                "reference_source_sha256": "sha-rttm",
                "replay_sha256": "sha-replay",
                "metrics": {"der": 0.46},
                "reference_speech_without_captions": {
                    "reference_speech_without_captions_seconds": 29.6,
                },
            }
            with patch("tools.ami_nemotron_live_score.score_case",
                       return_value=mock_audit):
                output = score_independent(root, "TS3003b", replay_path)
            self.assertEqual(output["independent_acoustic_metrics"]["der"], 0)
            self.assertEqual(output["published_caption_metrics"]["der"], 0.46)
            self.assertTrue(output["under_10pct_standard_der"])
            self.assertTrue(output["continuous_stream"])

    def test_solo_metric_explicitly_excludes_overlapped_reference(self):
        reference = [
            {"start_ms": 0, "end_ms": 2000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
        ]
        predicted = [
            {"start_ms": 0, "end_ms": 2000, "speaker_id": 1},
        ]
        solo = _solo_only(reference, predicted, duration_ms=2000)
        self.assertEqual(solo["der"], 0)
        self.assertEqual(solo["reference_speaker_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""Offline paired overlap projection never fabricates source-separated words."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evaluation.ami.ami_offline_display_overlap import offline_display_overlap_report


class OfflineOverlapReportTests(unittest.TestCase):
    def _fixture(self, path: Path) -> None:
        path.write_text(json.dumps({
            "delivery_matched": True, "server_closed": True,
            "errors": [], "flags": [], "source_audio_duration_s": 240.0,
            "final_cluster_diagnostics": {
                "anonymous_timeline": {
                    "truncated": False, "total_segments": 1,
                    "segments": [{
                        "segment_id": 1,
                        "start_ms": 0, "end_ms": 1000,
                        "published_speaker_id": 1,
                        "caption_projection_status": "split_verified",
                        "caption_projection_turns": [
                            {"start_ms": 0, "end_ms": 430,
                             "speaker_id": 1, "word_count": 2},
                            {"start_ms": 600, "end_ms": 1000,
                             "speaker_id": 2, "word_count": 2},
                        ],
                    }],
                },
            },
        }), encoding="utf-8")

    def test_two_activity_views_are_separately_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            replay = root / "replay.json"
            self._fixture(replay)
            acoustic = [
                {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
                {"start_ms": 600, "end_ms": 1000, "speaker_id": 2},
            ]
            with (
                patch(
                    "tools.evaluation.ami.ami_offline_display_overlap.score_case",
                    return_value={
                        "source_audio_sha256": "audio",
                        "reference_source_sha256": "original-rttm",
                    },
                ),
                patch(
                    "tools.evaluation.ami.ami_offline_display_overlap._validated_independent",
                    return_value=acoustic,
                ),
            ):
                result = offline_display_overlap_report(
                    root, "TS3003b", replay,
                )
        self.assertEqual(result["display_speaker_cue_splits"], 1)
        self.assertEqual(
            result["canonical_caption_activity"][
                "overlap_clock_underrepresented_ms"
            ], 400,
        )
        self.assertEqual(
            result["actual_ui_display_activity"][
                "overlap_clock_underrepresented_ms"
            ], 400,
        )
        self.assertEqual(
            result["difference_in_underrepresented_acoustic_clock_ms"], 0,
        )
        self.assertIn("not recovered words", result["note"])


if __name__ == "__main__":
    unittest.main()

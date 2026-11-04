"""Reference protocols must stay separate and never alter existing AMI scores."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ami_nemotron_reference_diagnostic import (
    alternate_score, forced_reference,
)


class ForcedReferenceProtocolTests(unittest.TestCase):
    def test_forced_rttm_preserves_talker_ids_and_excerpt_offset(self):
        rttm = "\n".join([
            "SPEAKER TS3003b 1 1499.500 1.200 <NA> <NA> TS3003b.A <NA> <NA>",
            "SPEAKER TS3003b 1 1500.800 0.900 <NA> <NA> TS3003b.D <NA> <NA>",
            "SPEAKER TS3003b 1 1740.100 1.000 <NA> <NA> TS3003b.B <NA> <NA>",
        ])
        turns = forced_reference(rttm, "TS3003b", 1500)
        self.assertEqual(turns, [
            {"start_ms": 0, "end_ms": 700, "speaker": "TS3003b.A"},
            {"start_ms": 800, "end_ms": 1700, "speaker": "TS3003b.D"},
        ])

    def test_an_unexpected_speaker_or_meeting_is_rejected(self):
        with self.assertRaises(ValueError):
            forced_reference(
                "SPEAKER XX 1 1501 2 <NA> <NA> XX.A <NA> <NA>",
                "TS3003b", 1500,
            )

    def test_original_independent_score_is_not_caption_based(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cases").mkdir()
            case = {"reference_turns": [
                {"start_ms": 0, "end_ms": 1000, "speaker": "human_a"},
                {"start_ms": 1000, "end_ms": 2000, "speaker": "human_b"},
            ]}
            (root / "cases" / "TS3003b.json").write_text(
                json.dumps(case), encoding="utf-8",
            )
            prediction = [
                {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
                {"start_ms": 1000, "end_ms": 2000, "speaker_id": 2},
            ]
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps({
                "server_ready": {"independent_diarization": "nemotron3"},
                "source_audio_duration_s": 240.0,
                "delivery_matched": True,
                "server_closed": True,
                "errors": [], "flags": [],
                "final_cluster_diagnostics": {
                    "anonymous_timeline": {
                        "truncated": False,
                        "total_segments": 1,
                        "segments": [{
                            "start_ms": 0, "end_ms": 1000,
                            "published_speaker_id": 1,
                        }],
                    },
                },
            }), encoding="utf-8")
            forced = (
                "SPEAKER TS3003b 1 1500.000 0.500 <NA> <NA> TS3003b.A <NA> <NA>\n"
                "SPEAKER TS3003b 1 1501.000 1.000 <NA> <NA> TS3003b.B <NA> <NA>"
            )
            with (
                patch(
                    "tools.ami_nemotron_reference_diagnostic.score_case",
                    return_value={
                        "source_audio_sha256": "audio",
                        "reference_source_sha256": "old-reference",
                        "metrics": {"der": 0.77},
                    },
                ),
                patch(
                    "tools.ami_nemotron_reference_diagnostic._validated_independent",
                    return_value=prediction,
                ),
                patch(
                    "tools.ami_nemotron_reference_diagnostic.fetch_forced_rttm",
                    return_value=(forced, "new-reference", root / "forced.rttm"),
                ),
            ):
                result = alternate_score(root, "TS3003b", replay_path)
            self.assertEqual(result["original_only_words_independent_der"], 0)
            self.assertEqual(result["original_only_words_caption_der"], 0.77)
            self.assertGreater(result["forced_aligned_independent_der"], 0)
            self.assertTrue(result["different_reference_annotations"])
            self.assertIn("forced_aligned_caption_metrics", result)
            self.assertGreater(
                result["forced_aligned_caption_der"],
                result["forced_aligned_independent_der"],
            )
            self.assertTrue(result["same_audio_and_prediction"])


if __name__ == "__main__":
    unittest.main()

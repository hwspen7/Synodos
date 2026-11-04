"""Paired five-way AMI ablation evidence cannot be fabricated or mixed."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.phase1a_compare import compare_ablation_inventory
from tools.ami_shadow10_prepare import checksum


MODES = [
    "content_first", "whisper_window_vad_bypass",
    "bounded_vad_gap_recovery", "preserve_verified_unaligned",
    "word_timestamp_bounds",
]


def fixture(root: Path):
    rows = []
    for index, variant in enumerate(MODES):
        suffix = f"20261104-{index:06}.json"
        replay = root / ("replay_TS3003b_nemotron_" + suffix)
        replay.write_text(json.dumps({
            "source_audio_sha256": "a" * 64,
            "reference_scoring": {"cer": .08, "wer": .17},
        }), encoding="utf-8")
        audit = root / ("audit_TS3003b_nemotron_" + suffix)
        audit.write_text(json.dumps({
            "schema": "meeting-ami-independent-nemotron-live-score-v1",
            "meeting": "TS3003b",
            "source_audio_sha256": "a" * 64,
            "reference_source_sha256": "b" * 64,
            "replay_sha256": checksum(replay),
            "independent_acoustic_metrics": {
                "der": .18 if index == 0 else .21
            },
        }), encoding="utf-8")
        rows.append({"variant": variant,
                     "independent_der_path": str(audit)})
    return {
        "schema": "meeting-phase1a-five-candidate-ablation-v1",
        "source_commit": "c" * 40,
        "reference_human_verified": True,
        "candidate_trials": rows,
    }


class Phase1ACompareTests(unittest.TestCase):
    def test_exactly_paired_ablation_and_real_criteria(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            report = compare_ablation_inventory(fixture(root), root=root)
            self.assertEqual(
                report["candidate_meets_both_real_meeting_targets"],
                "content_first",
            )
            self.assertFalse(report["may_auto_promote"])
            self.assertEqual(len(report["ranked_acoustic_variants"]), 5)

    def test_unreviewed_text_cannot_claim_cer_acceptance(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            value = fixture(root)
            value["reference_human_verified"] = False
            report = compare_ablation_inventory(value, root=root)
            self.assertIsNone(
                report["candidate_meets_both_real_meeting_targets"]
            )

    def test_mismatched_replay_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            value = fixture(root)
            report = Path(value["candidate_trials"][0]["independent_der_path"])
            data = json.loads(report.read_text(encoding="utf-8"))
            data["replay_sha256"] = "0" * 64
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paired"):
                compare_ablation_inventory(value, root=root)

    def test_missing_variant_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            value = fixture(root)
            value["candidate_trials"].pop()
            with self.assertRaisesRegex(ValueError, "five"):
                compare_ablation_inventory(value, root=root)


if __name__ == "__main__":
    unittest.main()

"""AMI paired replay controls must fail closed on ASR/diarization regressions."""
from __future__ import annotations

import copy
import unittest

from tools.ami_live_regression_gate import compare_live_audits


def audit():
    return {
        "meeting": "TS3003b",
        "timeline_complete": True,
        "source_audio_sha256": "audio-immutable",
        "reference_source_sha256": "rttm-immutable",
        "reference_kind": "only_words",
        "replay_sha256": "distinct-specific-replay",
        "frame_step_ms": 100,
        "collar_ms": 0,
        "metrics": {
            "der": 0.40534,
            "missed_speaker_time": 0.33947,
            "false_alarm_time": 0.01750,
            "confusion_time": 0.04836,
            "reference_speaker_seconds": 217.1,
        },
        "missed_speaker_time_breakdown": {
            "unidentified_transcript_s": 29.9,
            "without_transcript_s": 30.8,
        },
        "reference_speech_without_captions": {
            "reference_speech_without_captions_seconds": 29.9,
        },
    }


class PairedAmiGateTests(unittest.TestCase):
    def test_same_frozen_replay_is_non_regression_not_content_proof(self):
        base = audit()
        candidate = copy.deepcopy(base)
        candidate["replay_sha256"] = "second-replay"
        x = compare_live_audits(base, candidate)
        self.assertTrue(x["non_regression_pass"])
        self.assertTrue(x["gates"]["der_not_worse_by_over_0p5pp"])
        self.assertIn("NOT evidence of recovered true words", x["caveat"])

    def test_observed_content_retention_regression_must_fail(self):
        before = audit()
        after = copy.deepcopy(before)
        after["replay_sha256"] = "regressed-replay"
        after["metrics"].update({
            "der": 0.49332,
            "missed_speaker_time": 0.39659,
            "false_alarm_time": 0.04468,
            "confusion_time": 0.05205,
        })
        after["missed_speaker_time_breakdown"].update({
            "unidentified_transcript_s": 43.7,
            "without_transcript_s": 32.1,
        })
        after["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 29.8
        result = compare_live_audits(before, after)
        self.assertFalse(result["non_regression_pass"])
        self.assertEqual(result["measurements"]["der"]["delta"], 0.08798)
        self.assertEqual(
            result["measurements"]["unidentified_speaker_seconds"]["delta"],
            13.8,
        )
        self.assertEqual(
            result["measurements"]["without_transcript_speaker_seconds"]["delta"],
            1.3,
        )
        self.assertTrue(
            result["gates"]["no_caption_clock_time_not_worse_by_over_0p5s"]
        )
        self.assertFalse(
            result["gates"]["without_transcript_speaker_time_not_worse_by_over_0p5s"]
        )

    def test_no_change_is_not_quality_improvement(self):
        base = audit()
        candidate = copy.deepcopy(base)
        candidate["replay_sha256"] = "another-real-run"
        result = compare_live_audits(base, candidate)
        self.assertTrue(result["non_regression_pass"])
        self.assertFalse(result["diarization_coverage_improvement_pass"])

    def test_small_der_gain_with_missing_captions_is_not_improvement(self):
        base = audit()
        candidate = copy.deepcopy(base)
        candidate["replay_sha256"] = "small-der-gain"
        candidate["metrics"]["der"] = 0.402
        candidate["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 31.0
        result = compare_live_audits(base, candidate)
        self.assertFalse(result["diarization_coverage_improvement_pass"])

    def test_substantive_paired_gains_meet_diarization_coverage_gate(self):
        base = audit()
        candidate = copy.deepcopy(base)
        candidate["replay_sha256"] = "genuine-measured-gain"
        candidate["metrics"].update({
            "der": 0.39,
            "missed_speaker_time": 0.327,
            "false_alarm_time": 0.015,
            "confusion_time": 0.048,
        })
        candidate["missed_speaker_time_breakdown"][
            "without_transcript_s"
        ] = 28.0
        candidate["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 28.5
        result = compare_live_audits(base, candidate)
        self.assertTrue(result["diarization_coverage_improvement_pass"])
        self.assertTrue(result["non_regression_pass"])

    def test_speaker_count_and_overlap_regressions_are_guarded(self):
        before = audit()
        after = copy.deepcopy(before)
        after["replay_sha256"] = "new-acoustic-run"
        before["metrics"].update({
            "reference_speaker_count": 4,
            "speaker_count_absolute_error": 2,
            "overlap_all_speakers_recall": 0.20,
        })
        after["metrics"].update({
            "reference_speaker_count": 4,
            "speaker_count_absolute_error": 3,
            "overlap_all_speakers_recall": 0.0,
        })
        result = compare_live_audits(before, after)
        self.assertFalse(result["non_regression_pass"])
        self.assertFalse(result["diarization_coverage_improvement_pass"])
        self.assertFalse(result["gates"][
            "speaker_count_absolute_error_not_worse"
        ])
        self.assertFalse(result["gates"][
            "overlap_all_speakers_recall_not_worse"
        ])

    def test_matching_metrics_without_matching_audio_are_rejected(self):
        before, after = audit(), audit()
        after["source_audio_sha256"] = "another-audio"
        with self.assertRaisesRegex(ValueError, "source_audio_sha256"):
            compare_live_audits(before, after)
        after = audit()
        after["reference_source_sha256"] = "tampered-rttm"
        with self.assertRaisesRegex(ValueError, "reference_source_sha256"):
            compare_live_audits(before, after)
        after = audit()
        after["timeline_complete"] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            compare_live_audits(before, after)

    def test_missing_provenance_and_different_reference_duration_are_rejected(self):
        before, after = audit(), audit()
        after["replay_sha256"] = None
        with self.assertRaisesRegex(ValueError, "concrete replay"):
            compare_live_audits(before, after)
        after = audit()
        after["metrics"]["reference_speaker_seconds"] = 215.9
        with self.assertRaisesRegex(ValueError, "duration differs"):
            compare_live_audits(before, after)


if __name__ == "__main__":
    unittest.main()

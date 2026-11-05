"""A paired ASR coverage experiment is not an unconditional promotion."""
from __future__ import annotations

import copy
import unittest

from tools.phase1a_quality_pair import compare_pair


def evidence(cer=.15, uncapt=26.0, *, bypass=0, retry=0):
    audio = "a" * 64
    reference = "b" * 64
    return {
        "acoustic": {
            "schema": "meeting-ami-independent-nemotron-live-score-v1",
            "meeting": "TS3003b",
            "source_audio_sha256": audio,
            "reference_source_sha256": reference,
            "independent_acoustic_metrics": {"der": .23},
            "published_caption_metrics": {"der": .28},
        },
        "text": {
            "schema": "meeting-phase1-human-text-acceptance-v1",
            "source_audio_sha256": audio,
            "human_reference_sha256": "c" * 64,
            "gates": {
                "human_reference_confirmed": False,
                "complete_pcm_delivery": True,
            },
            "measurement": {
                "cer": cer,
                "wer": cer + .08,
                "reference_character_deletion_rate": cer - .04,
                "confirmed_text_changes": 0,
            },
        },
        "trace": {
            "schema": "meeting-ami-asr-stage-gap-audit-v1",
            "meeting": "TS3003b",
            "reference_protocol": "only_words",
            "source_audio_sha256": audio,
            "reference_source_sha256": reference,
            "reference_clock_no_caption_seconds": uncapt,
            "window_internal_vad_bypassed_cycles": bypass,
            "recovery_counts": {
                "recovery_attempted": retry,
                "recovery_confirmed": 0,
            },
        },
    }


class PairedCoverageTests(unittest.TestCase):
    def test_same_source_candidate_gain_is_still_not_a_release(self):
        result = compare_pair(
            evidence(),
            evidence(cer=.12, uncapt=18.0, bypass=200, retry=4),
        )
        self.assertTrue(result["same_audio_nonregression_screen"])
        self.assertTrue(result["same_audio_coverage_improvement_screen"])
        self.assertTrue(result["candidate_inner_window_vad_bypassed"])
        self.assertTrue(result["candidate_recovery_executed"])
        self.assertFalse(result["production_promotion_authorized"])
        self.assertFalse(result["human_review_complete"])

    def test_better_coverage_cannot_hide_worse_cer(self):
        result = compare_pair(
            evidence(), evidence(cer=.18, uncapt=20.0, bypass=50, retry=2),
        )
        self.assertFalse(result["same_audio_nonregression_screen"])
        self.assertFalse(result["same_audio_coverage_improvement_screen"])
        self.assertGreater(result["candidate_minus_baseline"]["cer"], 0)

    def test_mismatched_audio_or_text_reference_is_rejected(self):
        a, b = evidence(), evidence()
        b["acoustic"]["source_audio_sha256"] = "d" * 64
        b["text"]["source_audio_sha256"] = "d" * 64
        b["trace"]["source_audio_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Different source_audio_sha256"):
            compare_pair(a, b)
        b = evidence()
        b["text"]["human_reference_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "Different human-text"):
            compare_pair(a, b)

    def test_invalid_metric_and_unverified_replay_fail_closed(self):
        a, b = evidence(), evidence()
        b["text"]["measurement"]["cer"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Missing/invalid"):
            compare_pair(a, b)
        b = evidence()
        b["text"]["gates"]["complete_pcm_delivery"] = False
        result = compare_pair(a, b)
        self.assertFalse(result["same_audio_nonregression_screen"])


if __name__ == "__main__":
    unittest.main()

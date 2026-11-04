"""Quality-first speech recovery and fail-closed AMI promotion contracts."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speech.contracts import Utterance
from speech.coverage_retention import choose_safe_drop
from speech.gap_recovery import recover_missing_voice, uncovered_voice_intervals
from speech.quality_guard import verify_utterances
from tools.ami_quality_experiments import (
    CONTROL_ENV, PROFILES, profile_environment, validate_profiles,
)
from tools.ami_quality_gate import assess_quality_gain


def vad(start_ms, end_ms):
    return ({"start": start_ms * 16, "end": end_ms * 16},)


def baseline():
    return {
        "meeting": "TS3003b",
        "timeline_complete": True,
        "source_audio_sha256": "same-audio",
        "reference_source_sha256": "same-rttm",
        "reference_kind": "only_words",
        "replay_sha256": "old",
        "collar_ms": 0,
        "frame_step_ms": 100,
        "metrics": {
            "der": 0.40534,
            "missed_speaker_time": 0.33947,
            "false_alarm_time": 0.0175,
            "confusion_time": 0.04836,
            "reference_speaker_seconds": 217.1,
            "speaker_count_absolute_error": 2,
            "overlap_all_speakers_recall": 0.0,
        },
        "missed_speaker_time_breakdown": {
            "unidentified_transcript_s": 29.9,
            "without_transcript_s": 30.8,
        },
        "reference_speech_without_captions": {
            "reference_speech_without_captions_seconds": 29.9,
        },
    }


class ShortVoiceRecoveryTests(unittest.TestCase):
    def test_short_vad_interval_is_eligible_only_with_lower_threshold(self):
        self.assertEqual(
            uncovered_voice_intervals(vad(400, 1050), (), 3000), ()
        )
        self.assertEqual(
            uncovered_voice_intervals(
                vad(400, 1050), (), 3000, min_uncovered_ms=450,
            ), ((400, 1050),)
        )

    def test_real_second_decode_recovers_subsecond_utterance(self):
        audio = np.full(3 * 16000, 0.035, dtype=np.float32)
        calls = []

        def infer(clip, **kw):
            calls.append(len(clip))
            return [SimpleNamespace(
                start=0.18, end=0.73, text="Yes, agreed.",
                words=[], no_speech_prob=0.05,
                avg_logprob=-0.1, compression_ratio=1.05,
            )], SimpleNamespace(language="en")
        original = (Utterance("Next point.", 1700, 2700),)
        recovered, totals = recover_missing_voice(
            audio, vad(400, 1050), original,
            infer, {"vad_filter": True}, language="en",
            min_uncovered_ms=450,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(totals["recovery_confirmed"], 1)
        self.assertEqual(recovered[0].recognition_origin, "gap_recovery")
        self.assertEqual(recovered[1], original[0])
        self.assertGreaterEqual(recovered[0].start_ms, 400)
        self.assertLessEqual(recovered[0].end_ms, 1050 + 180)

    def test_bounded_reclaim_keeps_untranscribed_short_voice(self):
        args = dict(
            proposed_drop_samples=2500 * 16,
            audio_samples=4000 * 16,
            sample_rate=16000,
            global_start_samples=0,
            speech_intervals_ms=((400, 1050),),
            utterances=(Utterance("Later", 1700, 2500),),
            attempts={},
        )
        baseline_drop, _ = choose_safe_drop(**args)
        candidate_drop, _ = choose_safe_drop(
            **{**args, "attempts": {}, "min_uncovered_ms": 450}
        )
        self.assertEqual(baseline_drop, 2500 * 16)
        self.assertEqual(candidate_drop, (400 - 180) * 16)

    def test_short_vad_backed_acknowledgement_preserved_in_candidate(self):
        segment = SimpleNamespace(
            start=0.1, end=0.45, text="Yep.", words=[],
            no_speech_prob=0.01, avg_logprob=-0.1,
            compression_ratio=1.1,
        )
        regions = vad(200, 340)
        self.assertEqual(verify_utterances([segment], regions, 1000, "en"), ())
        candidate = verify_utterances(
            [segment], regions, 1000, "en", short_voice_support_ms=100,
        )
        self.assertEqual([item.text for item in candidate], ["Yep."])
        # A long piece containing only a tiny voiced blip stays rejected.
        long_piece = SimpleNamespace(
            start=0.0, end=6.0, text="Possible hallucination", words=[],
            no_speech_prob=0.05, avg_logprob=-0.1,
            compression_ratio=1.1,
        )
        self.assertEqual(verify_utterances(
            [long_piece], regions, 7000, "en", short_voice_support_ms=100,
        ), ())

    def test_bad_short_gap_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            recover_missing_voice(
                np.zeros(16000), vad(0, 750), (),
                lambda a, **kw: ([], SimpleNamespace(language="en")),
                {}, language="en", min_uncovered_ms=100,
            )


class A_BQualityGateTests(unittest.TestCase):
    def test_observed_20261104_result_is_not_an_improvement(self):
        before = baseline()
        after = copy.deepcopy(before)
        after["replay_sha256"] = "20261104-test"
        after["metrics"].update({
            "der": 0.4058, "missed_speaker_time": 0.33625,
            "false_alarm_time": 0.01796,
            "confusion_time": 0.05159,
        })
        after["missed_speaker_time_breakdown"].update({
            "unidentified_transcript_s": 26.7,
            "without_transcript_s": 32.7,
        })
        after["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 31.3
        decision = assess_quality_gain(before, after)
        self.assertFalse(decision["measured_gain_on_this_case"])
        self.assertFalse(
            decision["checks"]["no_caption_clock_improved_by_0p5s"]
        )
        self.assertFalse(
            decision["checks"]["der_improved_by_at_least_0p2pp"]
        )

    def test_frozen_measured_gain_can_pass_without_auto_promotion(self):
        before = baseline()
        after = copy.deepcopy(before)
        after["replay_sha256"] = "real-other-replay"
        after["metrics"].update({
            "der": 0.375, "missed_speaker_time": 0.31,
            "false_alarm_time": 0.0171,
            "confusion_time": 0.0479,
            "speaker_count_absolute_error": 1,
            "overlap_all_speakers_recall": 0.05,
        })
        after["missed_speaker_time_breakdown"].update({
            "unidentified_transcript_s": 24.5,
            "without_transcript_s": 27.4,
        })
        after["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 26.7
        decision = assess_quality_gain(before, after)
        self.assertTrue(decision["measured_gain_on_this_case"])
        self.assertIn("Single-case improvement", decision["limits"])

    def test_worse_confusion_and_different_audio_cannot_pass(self):
        before = baseline()
        after = copy.deepcopy(before)
        after["replay_sha256"] = "other-run"
        after["metrics"].update({
            "der": 0.39, "missed_speaker_time": 0.31,
            "confusion_time": 0.055,
        })
        after["missed_speaker_time_breakdown"]["without_transcript_s"] = 26
        after["reference_speech_without_captions"][
            "reference_speech_without_captions_seconds"
        ] = 24
        self.assertFalse(
            assess_quality_gain(before, after)["measured_gain_on_this_case"]
        )
        after["source_audio_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "source_audio_sha256"):
            assess_quality_gain(before, after)

    def test_profiles_are_isolated_and_never_force_four_speakers(self):
        inherited = {
            "MEETING_WHISPER_GAP_RECOVERY": "1",
            "MEETING_SPEAKER_CLUSTER_STRATEGY": "strict",
            "CUSTOM_UNRELATED_VALUE": "kept",
        }
        before = dict(inherited)
        control = profile_environment(inherited, "control")
        candidate = profile_environment(inherited, "speaker_candidate")
        self.assertEqual(inherited, before)
        self.assertEqual(control["MEETING_WHISPER_GAP_RECOVERY"], "0")
        self.assertEqual(control["MEETING_SPEAKER_CLUSTER_STRATEGY"], "bootstrap")
        self.assertEqual(candidate["MEETING_WHISPER_GAP_RECOVERY"], "1")
        self.assertEqual(candidate["MEETING_SPEAKER_CLUSTER_STRATEGY"], "candidate")
        self.assertEqual(candidate["MEETING_SPEAKER_PROBE_MIN_S"], "0.9")
        self.assertEqual(candidate["CUSTOM_UNRELATED_VALUE"], "kept")
        self.assertNotIn("MEETING_ORACLE_SPEAKER_COUNT", candidate)
        self.assertEqual(
            validate_profiles("recovery,single_vad,speaker_candidate"),
            ["recovery", "single_vad", "speaker_candidate"],
        )
        with self.assertRaises(ValueError):
            validate_profiles("recovery,recovery")
        with self.assertRaises(ValueError):
            profile_environment({}, "unknown")
        self.assertEqual(CONTROL_ENV["MEETING_DIARIZATION_DIAGNOSTICS"], "1")
        self.assertIn("control", PROFILES)


if __name__ == "__main__":
    unittest.main()

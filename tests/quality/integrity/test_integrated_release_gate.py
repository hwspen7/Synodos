"""All Phase 1A/1B/2 gate items are required before Phase 3."""
import copy
import unittest

from tools.validation.integrated_release_gate import assess, REQUIRED_ABLATIONS, REQUIRED_CI


COMMIT = "a" * 40


def evidence():
    lang = {
        "schema": "meeting-phase1-ten-language-text-gate-v1",
        "pass": True,
        "supported_language_codes": [
            "en", "zh", "fr", "es", "ar", "id", "ja", "ru", "de", "hi",
        ],
        "selected_reference_case_count": 110,
        "overall": {"cer": 0.05078},
        "gates": {
            "exactly_ten_supported_languages": True,
            "exactly_110_frozen_reference_clips": True,
            "all_selected_alignments_evaluable": True,
            "no_empty_predictions": True,
            "weighted_cer_within_limit": True,
            "weighted_deletion_within_limit": True,
            "each_supported_language_within_limit": True,
        }
    }
    human = {
        "schema": "meeting-phase1-human-text-acceptance-v1",
        "pass": True, "source_audio_sha256": "b" * 64,
        "gates": {
            "human_reference_confirmed": True,
            "confirmed_text_not_silently_revised": True,
            "confirmed_segments_never_disappeared": True,
            "reference_deletion_at_most_threshold": True,
            "character_error_at_most_threshold": True,
        },
    }
    acoustic = {
        "schema": "meeting-ami-independent-nemotron-live-score-v1",
        "meeting": "TS3003b", "source_audio_sha256": "b" * 64,
        "independent_acoustic_metrics": {"der": .19},
        "under_20pct_standard_der": True,
    }
    comparison = {
        "schema": "meeting-phase1b-paired-comparison-v1",
        "complete_heldout_cohort": True,
        "paired_meeting_count": 10, "acoustic_goal_met": True,
        "may_auto_promote_to_live": False,
    }
    qwen = {
        "schema": "meeting-phase1b-qwen-asr-shadow-v1",
        "infer_enabled": True,
        "reference_words_used_to_select_windows": False,
        "auto_replaced_live_text": False,
        "detected_candidate_windows": 2,
        "candidate_transcripts": [{"candidate_text": "candidate"}],
    }
    identity = {
        "schema": "meeting-phase2-cross-session-acceptance-v1",
        "holdout_goal_met": True,
        "dev_pairs": 40, "holdout_pairs": 40,
        "may_auto_update_production_threshold": False,
    }
    ci = {
        "schema": "meeting-phase1-github-actions-diagnostic-v1",
        "commit_sha": COMMIT,
        "all_required_checks_green": True,
        "checks": [
            {
                "name": name, "status": "completed",
                "conclusion": "success", "github_actions_job": True,
                "executed_steps": 10,
            }
            for name in REQUIRED_CI
        ],
    }
    ablation = {
        "schema": "meeting-phase1a-five-candidate-ablation-v1",
        "source_commit": COMMIT,
        "candidate_trials": [
            {"variant": item, "independent_der_path": f"private/{item}.json"}
            for item in REQUIRED_ABLATIONS
        ]
    }
    return (lang, human, acoustic, comparison, qwen, identity, ci, ablation)


class ReleaseContractTests(unittest.TestCase):
    def test_complete_synthetic_evidence_meets_schema(self):
        result = assess(*evidence(), source_commit=COMMIT)
        self.assertTrue(result["pass"])
        self.assertEqual(result["next_stage"], "phase3")
        self.assertFalse(result["automatic_merge_or_tag"])

    def test_actual_unaccepted_ami_values_stay_blocked(self):
        rows = list(copy.deepcopy(evidence()))
        rows[1]["pass"] = False
        rows[1]["gates"]["character_error_at_most_threshold"] = False
        rows[2]["independent_acoustic_metrics"]["der"] = .23768
        rows[2]["under_20pct_standard_der"] = False
        result = assess(*rows, source_commit=COMMIT)
        self.assertFalse(result["pass"])
        self.assertFalse(result["gates"]["same_audio_der_under_20_percent"])
        self.assertEqual(result["next_stage"], "phase1_2_blocked")

    def test_language_cer_between_ten_and_fifteen_does_not_pass(self):
        rows = list(evidence())
        rows[0]["overall"]["cer"] = 0.12
        result = assess(*rows, source_commit=COMMIT)
        self.assertFalse(result["pass"])
        self.assertFalse(result["gates"]["ten_language_frozen_cer"])

    def test_missing_one_paired_meeting_or_ci_job_blocks(self):
        rows = list(evidence())
        rows[3]["paired_meeting_count"] = 9
        rows[6]["checks"] = rows[6]["checks"][:-1]
        out = assess(*rows, source_commit=COMMIT)
        self.assertFalse(out["gates"]["three_model_paired_ten_meeting_comparison"])
        self.assertFalse(out["gates"]["hosted_checks_for_exact_commit"])

    def test_mismatch_audio_sha_and_unreviewed_text_blocks(self):
        rows = list(evidence())
        rows[1]["gates"]["human_reference_confirmed"] = False
        rows[2]["source_audio_sha256"] = "c" * 64
        out = assess(*rows, source_commit=COMMIT)
        self.assertFalse(out["gates"]["same_audio_der_under_20_percent"])
        self.assertFalse(out["gates"]["reviewed_real_meeting_text"])


if __name__ == "__main__":
    unittest.main()

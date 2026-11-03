"""Deterministic no-download contract tests for the Fixed150 preparation and scoring.

These tests do not imply that the 150 GB/MB-scale source files or actual GPU
models have run. The real inference gate stays blocked until local replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import wave

from tools.fixed150_prepare import (
    AMI_ROWS, FLEURS_LOCALES, FLEURS_ROWS, LAION_ROWS,
    MIND_LOCALES, MIND_ROWS, ami_spans, checksum, verify_case, verify_lock,
    choose_source_file,
)
from phase07.quality import score_pair, aggregate
from speech.qwen import qwen_language_name, LANGUAGES
from speech.language import normalize_language
from tools.fixed150_ablation import choose_cases, likely_failure
from tools.fixed150_reference_audit import source_flags, evaluate as audit_references, FOCUS_15
from tools.fixed150_system_audit import summarize as summarize_system_ablation
from tools.fixed150_system_triage import inspect as inspect_system_triage
from tools.fixed150_stage_audit import select_impacted, measure as measure_stage
from speech.audio_windows import speech_windows
from speech.text_format import join_transcript_text
from tools.fixed150_dashboard import (
    summarize_multis, summarize_singles, token_recall, token_bag,
)


class Fixed150Contracts(unittest.TestCase):
    def test_fleurs_pinned_revision_parquet_data_layout(self):
        # The current google/fleurs revision 70bb2e8 stores files under
        # parquet-data/<locale>/, not at the dataset root.
        available = {
            "parquet-data/ar_eg/validation-00000-of-00001.parquet",
            "parquet-data/hi_in/validation-00000-of-00001.parquet",
            "data/ar_eg/metadata.tsv",
        }
        self.assertEqual(
            choose_source_file(available, "ar_eg/validation-00000-of-00001.parquet",
                               repo="google/fleurs", revision="70bb2e8"),
            "parquet-data/ar_eg/validation-00000-of-00001.parquet",
        )

    def test_missing_or_ambiguous_repo_file_fails_closed(self):
        target = "data/train-00000-of-00001.parquet"
        self.assertEqual(
            choose_source_file({"train-00000-of-00001.parquet"}, target),
            "train-00000-of-00001.parquet",
        )
        with self.assertRaises(FileNotFoundError):
            choose_source_file({
                "a/train-00000-of-00001.parquet",
                "b/train-00000-of-00001.parquet",
            }, target)

    def test_thai_token_metric_is_not_fabricated(self):
        row = score_pair("ภาษาไทยทดสอบ", "ภาษาไทย", language="th")
        self.assertIsNone(row["mixed_token_error_rate"])
        self.assertFalse(row["token_metric_supported"])
        self.assertGreater(row["reference_characters"], 0)
        row2 = score_pair("hello world", "hello", language="en")
        self.assertTrue(row2["token_metric_supported"])
        self.assertGreater(row2["mixed_token_error_rate"], 0)
        summary = aggregate([row])
        self.assertIsNone(summary["mixed_token_error_rate"])
        self.assertIsNotNone(summary["cer"])

    def test_existing_historical_thai_report_can_render_without_reinference(self):
        fake = {"cases": [
            {"id": f"fixed150-{i:03d}", "kind": "single",
             "language": ("th" if i in (97, 98, 99, 100) else "en"),
             "scenario": "read_speech"} for i in range(1, 101)
        ]}
        rows = []
        for i in range(100):
            rows.append({
                "index": i, "reference_characters": 10, "character_edits": 2,
                "reference_tokens": 1, "token_edits": 3 if i >= 96 else 0,
                "cer": 0.2, "mixed_token_error_rate": 3.0 if i >= 96 else 0,
                "expected_language": "th" if i >= 96 else "en",
                "predicted_language": "th" if i >= 96 else "en",
                "latency_ms": 100,
            })
        fake_report = {"infer": True, "samples": 100,
                       "engines": {"whisper": {"samples": rows}}}
        scored = summarize_singles(fake, fake_report)
        data = scored["engines"]["whisper"]
        self.assertEqual(scored["status"], "MEASURED")
        self.assertEqual(data["mixed_token_er"], 0)
        self.assertIsNone(next(x["mixed_token_er"] for x in data["per_language"]
                               if x["language"] == "th"))

    def test_qwen_language_mapping_uses_supported_names(self):
        self.assertIsNone(qwen_language_name("auto"))
        self.assertIsNone(qwen_language_name("und"))
        for code, expected in {
            "ar": "Arabic", "hi": "Hindi", "id": "Indonesian",
            "tr": "Turkish", "vi": "Vietnamese", "ko": "Korean",
            "fa": "Persian", "el": "Greek", "fil": "Filipino",
        }.items():
            self.assertEqual(qwen_language_name(code), expected)
            self.assertEqual(normalize_language(expected), code)
        self.assertEqual(len(LANGUAGES), 30)
        with self.assertRaises(ValueError):
            qwen_language_name("bn")
        with self.assertRaises(ValueError):
            qwen_language_name("ur")

    def test_targeted_ablation_selection_is_stable(self):
        fake = {"cases": [
            {"id": "single-1", "kind": "single", "language": "ko"},
            {"id": "single-2", "kind": "single", "language": "ko"},
            {"id": "single-3", "kind": "single", "language": "pt"},
            {"id": "multi-4", "kind": "multi", "language": "ko"},
        ]}
        self.assertEqual(
            [x["id"] for x in choose_cases(fake, {"ko", "pt"}, 1)],
            ["single-1", "single-3"],
        )
        with self.assertRaises(ValueError):
            choose_cases(fake, {"ur"}, 1)
        self.assertIn("PIPELINE_LOSS_OR_SEGMENTATION",
                      likely_failure(.8, .2, .2, "ko", "ko"))
        self.assertIn("AUTO_LANGUAGE_OR_DECODING",
                      likely_failure(.8, .8, .1, "en", "ko"))

    def test_korean_45sec_reference_audit_flags_missing_coverage_risk(self):
        ref = ("네 안녕하세요 외국에서 카드를 사용하고 싶은데 "
               "미국에 있을 때 사용이 가능한지 확인해 줄 수 있나 해서요")
        issues = source_flags(45.0, ref)
        self.assertIn("LONG_AUDIO_SHORT_REFERENCE", issues)
        self.assertIn("LOW_REFERENCE_CHARACTER_DENSITY", issues)
        self.assertFalse(source_flags(6.0, ref))

    def test_annotation_audit_keeps_flagged_samples_in_raw_denominator(self):
        sources = [
            {"id": f"fixed150-{i:03d}", "kind": "single",
             "language": "ko" if i == 36 else "en",
             "duration_s": 45.0 if i == 36 else 6.0,
             "reference": "가" * 43 if i == 36 else "hello how are you"}
            for i in range(1, 101)
        ]
        rows = [
            {"index": i, "reference_characters": 43 if i == 35 else 14,
             "character_edits": 91 if i == 35 else 0,
             "cer": 91 / 43 if i == 35 else 0}
            for i in range(100)
        ]
        report = audit_references(
            {"cases": sources},
            {"infer": True, "samples": 100, "engines": {
                "whisper": {"samples": rows}}},
            ("ko", "en"),
        )
        self.assertEqual(report["selected_samples"], 100)
        self.assertGreater(report["raw_selected_cer"], 0)
        ko = next(x for x in report["per_language"] if x["language"] == "ko")
        self.assertEqual(ko["flagged_for_review"], 1)
        self.assertIsNone(ko["sensitivity_cer_excluding_flagged"])
        self.assertFalse(report["release_certified"])

    def test_reference_audit_default_scope_is_15(self):
        self.assertEqual(len(FOCUS_15), 15)
        self.assertEqual(len(set(FOCUS_15)), 15)

    def test_korean_word_boundaries_are_preserved(self):
        self.assertEqual(join_transcript_text("안녕하세요", " 여러분"),
                         "안녕하세요 여러분")
        self.assertEqual(join_transcript_text("안녕하세요", "여러분"),
                         "안녕하세요여러분")
        self.assertEqual(join_transcript_text("\u4f60\u597d", "\u4e16\u754c"), "\u4f60\u597d\u4e16\u754c")
        self.assertEqual(join_transcript_text("Hello", " world"), "Hello world")
        self.assertEqual(join_transcript_text("Hello", " !"), "Hello!")

    def test_systemwide_triage_marks_candidate_stages_without_guessing_cause(self):
        sample = {
            "schema": "meeting-fixed150-system-ablation-v1",
            "cases": {
                "fixed150-048": {
                    "id": "fixed150-048", "language": "pt",
                    "duration_s": 61.1,
                    "acoustic_geometry": {
                        "asr_windows": 4,
                        "window_boundary_inside_detected_speech": 2,
                    },
                    "reference_review_flags": [],
                    "production_windowed": {"cer": .798},
                    "production_whole_clip": {"cer": .40},
                    "direct_whisper": {"cer": .284},
                },
                "fixed150-011": {
                    "id": "fixed150-011", "language": "en",
                    "duration_s": 4.5,
                    "acoustic_geometry": {"asr_windows": 1},
                    "reference_review_flags": [],
                    "production_windowed": {"cer": 0},
                    "production_whole_clip": {"cer": 0},
                    "direct_whisper": {"cer": 0},
                },
            },
        }
        triage = inspect_system_triage(sample)
        self.assertEqual(triage["cases"], 2)
        self.assertEqual(triage["cases_harmed_by_windowed_vs_direct"], 1)
        self.assertEqual(triage["per_language_affected"], {"pt": 1})
        top = triage["ranked"][0]
        self.assertEqual(top["id"], "fixed150-048")
        self.assertIn("WINDOW_ROUTING_CANDIDATE", top["flags"])
        self.assertIn("POSTPROCESS_OR_DECODING_DIFFERENCE_CANDIDATE", top["flags"])
        self.assertIn("WINDOW_SEAM_WITHIN_DETECTED_SPEECH", top["flags"])
        with self.assertRaises(ValueError):
            inspect_system_triage({"schema": "bad", "cases": {}})

    def test_stage_audit_selects_only_verified_impacted_originals(self):
        cases = [
            {"id": f"fixed150-{i:03d}", "kind": "single",
             "language": "pt" if i == 48 else "en",
             "sha256": f"digest{i}"}
            for i in range(1, 101)
        ]
        report = {
            "schema": "meeting-fixed150-system-ablation-v1",
            "cases": {
                case["id"]: {
                    "id": case["id"],
                    "sha256": case["sha256"],
                    "production_windowed": {
                        "cer": .798 if case["id"] == "fixed150-048" else .10,
                    },
                    "direct_whisper": {"cer": .284 if case["id"] == "fixed150-048" else .10},
                } for case in cases
            },
        }
        found = select_impacted({"cases": cases}, report)
        self.assertEqual([x["id"] for x in found], ["fixed150-048"])
        report["cases"]["fixed150-048"]["sha256"] = "corrupt"
        with self.assertRaises(ValueError):
            select_impacted({"cases": cases}, report)
        self.assertEqual(measure_stage("abc", "abc", "en")["cer"], 0)

    def test_continuous_speech_is_hard_cut_at_18_seconds(self):
        samples = 40000 * 16
        windows = speech_windows(
            [{"start": 0, "end": samples}],
            samples, sample_rate=16000, max_window_s=18,
            pad_ms=0,
        )
        self.assertEqual(windows, [
            (0, 18 * 16000), (18 * 16000, 36 * 16000),
            (36 * 16000, samples),
        ])
        # 18s and 36s are not validated pauses; they bisect one VAD region.

    def test_system_ablation_paired_scores_keep_improvements_and_harm(self):
        def row(cid, lang, window, whole, raw, flagged=False):
            return {
                "id": cid, "language": lang, "reference_review_flags":
                    ["LOW_REFERENCE_CHARACTER_DENSITY"] if flagged else [],
                "production_windowed": {
                    "reference_characters": 100, "character_edits": window,
                },
                "production_whole_clip": {
                    "reference_characters": 100, "character_edits": whole,
                },
                "direct_whisper": {
                    "reference_characters": 100, "character_edits": raw,
                },
                "acoustic_geometry": {
                    "window_boundary_inside_detected_speech": 1 if flagged else 0,
                },
                "delta": {
                    "window_minus_global_pp": window - whole,
                    "window_minus_raw_pp": window - raw,
                },
            }
        result = summarize_system_ablation([
            row("one", "pt", 80, 35, 28, flagged=True),
            row("two", "en", 0, 12, 8),
        ], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["cases_window_worse_than_raw_5pp"], 1)
        self.assertEqual(result["cases_window_better_than_raw_5pp"], 1)
        self.assertEqual(result["cases_window_worse_than_whole_5pp"], 1)
        self.assertEqual(result["cases_with_speech_boundary_cuts"], 1)
        self.assertEqual(result["cases_requiring_reference_review"], 1)
        self.assertAlmostEqual(result["overall_cer"]["production_windowed"], .40)
        self.assertEqual(result["per_language"]["pt"]["window_worse_than_raw_5pp"], 1)

    def test_exact_diverse_allocations(self):
        self.assertEqual(len(MIND_LOCALES) * len(MIND_ROWS), 56)
        self.assertEqual(len(FLEURS_LOCALES) * len(FLEURS_ROWS), 44)
        self.assertEqual(len(AMI_ROWS), 25)
        self.assertEqual(len(LAION_ROWS), 25)
        langs = {code for _, code in MIND_LOCALES + FLEURS_LOCALES}
        self.assertGreaterEqual(len(langs), 20)
        self.assertEqual(len(set(AMI_ROWS)), 25)
        self.assertEqual(len(set(LAION_ROWS)), 25)

    def test_real_ami_timestamp_token_decoder(self):
        text = "<|0.00|> Hello there.<|1.42|><|1.42|> Um.<|2.72|>"
        spans = ami_spans(text, "speaker_1", 3.0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["start_ms"], 0)
        self.assertEqual(spans[1]["end_ms"], 2720)
        self.assertEqual(spans[1]["text"], "Um.")

    def test_timestamp_rejects_out_of_bounds(self):
        self.assertEqual(ami_spans("<|4.00|> Bad.<|8.00|>", "s1", 3.0), [])

    def test_unmeasured_results_not_zero_error(self):
        fake = {"cases": [
            {"id": f"fixed150-{i:03d}", "kind": "single",
             "language": "en", "scenario": "read_speech"} for i in range(1, 101)
        ]}
        self.assertEqual(summarize_singles(fake, None)["status"], "NOT RUN")
        missing = {"infer": True, "samples": 100, "engines": {
            "whisper": {"samples": [{"index": i, "error": "GPU_NOT_READY"}
                                    for i in range(100)]}
        }}
        result = summarize_singles(fake, missing)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["engines"]["whisper"]["valid"], 0)

    def test_token_recall_does_not_confuse_missing_voice_with_success(self):
        self.assertAlmostEqual(token_recall("fresh talking audio", "fresh audio"), 2/3)
        self.assertEqual(token_recall("fresh talking audio", ""), 0)
        self.assertEqual(token_recall("a I", "a I"), 1.0)
        self.assertEqual(token_recall("I no OK uh", "I no"), 0.5)
        self.assertEqual(token_recall("no", ""), 0.0)
        self.assertTrue(token_bag("Talk talk")["talk"] == 2)

    def test_source_sha_is_not_trusted_without_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "audio").mkdir()
            wav = root / "audio" / "x.wav"
            with wave.open(str(wav), "wb") as out:
                out.setframerate(16000)
                out.setnchannels(1)
                out.setsampwidth(2)
                out.writeframes(b"\0\0" * 16000)
            case = {"id": "x", "audio": "audio/x.wav", "sha256": checksum(wav)}
            verify_case(root, case)
            wav.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                verify_case(root, case)

    def test_partial_or_missing_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "lock.json").write_text(json.dumps({
                "schema": "meeting-fixed150-v1", "cases": []
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_lock(root)

    def test_multispeaker_without_replay_remains_blocked(self):
        fake = {"cases": [
            {"id": f"fixed150-{i:03d}", "kind": "multi",
             "scenario": "real_meeting_two_speakers"}
            for i in range(101, 151)
        ]}
        result = summarize_multis(fake, None)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["valid"], 0)
        self.assertEqual(result["groups"]["real_meeting_two_speakers"]["count"], 50)

    def test_speaker_permutation_invariant_content_attribution(self):
        case = {
            "id": "fixed150-101", "kind": "multi", "scenario": "real_meeting_two_speakers",
            "sha256": "frozen",
            "annotation_precision": "utterance",
            "speakers": [
                {"speaker": "speaker_1", "reference": "orange meeting notebook"},
                {"speaker": "speaker_2", "reference": "green transport bus"},
            ],
            "reference_segments": [
                {"speaker": "speaker_1", "start_ms": 0, "end_ms": 1000},
                {"speaker": "speaker_2", "start_ms": 1000, "end_ms": 2000},
            ],
        }
        prediction = {"results": {"fixed150-101": {
            "status": "completed", "sha256": "frozen",
            "sent_samples": 32000, "acknowledged_samples": 32000,
            "segments": [
                {"start_ms": 0, "end_ms": 1000,
                 "speaker_id": "B", "text": "orange meeting notebook"},
                {"start_ms": 1000, "end_ms": 2000,
                 "speaker_id": "A", "text": "green transport bus"},
            ],
        }}}
        report = summarize_multis({"cases": [case]}, prediction)
        self.assertEqual(report["cases"][0]["der"], 0)
        self.assertEqual(report["cases"][0]["attributed_content_recall"], [1, 1])


if __name__ == "__main__":
    unittest.main()

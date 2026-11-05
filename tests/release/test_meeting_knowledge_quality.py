"""Deterministic gates only; real audio is never simulated as passing here."""
from __future__ import annotations

import copy
import unittest

from tools.meeting_knowledge_contract import (
    FREEZE_SCHEMA, LEGACY_SYNTHETIC_SCHEMA, strict_synthetic, validate_manifest, score_answerable,
    score_negative, real_summary, release_decision,
)

H = "b" * 64


def make_synthetic(mode):
    return {
        "schema": "meeting-knowledge-synthetic-evaluation", "mode": mode,
        "passed": True, "real_ollama": True,
        "real_audio_gold_coverage": False, "human_semantic_truth_review": False,
        "text_model": "Saluki-27B", "embedding_model": "qwen3-embedding:0.6b" if mode == "hybrid" else None,
        "counts": {"answerable_pass": 36, "answerable_total": 36,
                   "retrieval_recall_at_8": 36, "abstention_pass": 6, "abstention_total": 6},
        "cases": [dict(kind="answerable", passed=True, retrieval_expected_in_top8=True) for _ in range(36)] +
                 [dict(kind="unanswerable", passed=True) for _ in range(6)],
    }


def make_manifest():
    meetings = [dict(meeting_id=f"m{i}", audio_sha256=H,
                     audio_file=f"E:\\recordings\\m{i}.wav",
                     audio_reviewed=True, synthetic_audio=False) for i in range(3)]
    cases = [dict(id=f"p{i}", kind="answerable", meeting_id=f"m{i % 3}",
                  language="en", question=f"What happened {i}?",
                  verified_from_audio=True, expected_text="verified fact",
                  gold_segment_ids=[i + 1], gold_interval_ms=[10, 100])
             for i in range(20)]
    cases += [dict(id=f"n{i}", kind=("cross_scope" if i == 0 else
                       "contradiction" if i == 1 else "unanswerable"),
                   meeting_id=f"m{i % 3}", language="en",
                   question=f"What is not known {i}?", verified_from_audio=True,
                   **({"foreign_meeting_id": "m1", "foreign_expected_text": "foreign verified fact"} if i == 0 else
                       {"contradiction_segment_ids": [11, 12]} if i == 1 else {}))
              for i in range(6)]
    return {"schema": "meeting-audio-review-gold",
            "corpus_type": "real_recorded_meetings",
            "gold_source": "independent_human_audio_review",
            "reviewer_id": "human-reviewer-1", "meetings": meetings, "cases": cases}


class SyntheticFreezeContract(unittest.TestCase):
    def test_accepted_current_reports(self):
        strict_synthetic(make_synthetic("lexical"), "lexical")
        strict_synthetic(make_synthetic("hybrid"), "hybrid")

    def test_historical_reports_remain_valid(self):
        report = make_synthetic("lexical")
        report["schema"] = LEGACY_SYNTHETIC_SCHEMA
        self.assertTrue(strict_synthetic(report, "lexical"))

    def test_reject_per_case_fake_pass(self):
        x = make_synthetic("lexical")
        x["cases"][4]["passed"] = False
        with self.assertRaises(ValueError):
            strict_synthetic(x, "lexical")

    def test_reject_missing_embedding_model(self):
        x = make_synthetic("hybrid")
        x["embedding_model"] = None
        with self.assertRaises(ValueError):
            strict_synthetic(x, "hybrid")

    def test_reject_fake_real_audio_claim(self):
        x = make_synthetic("lexical")
        x["real_audio_gold_coverage"] = True
        with self.assertRaises(ValueError):
            strict_synthetic(x, "lexical")

    def test_reject_boolean_counts(self):
        x = make_synthetic("lexical")
        x["counts"]["answerable_pass"] = True
        with self.assertRaises(ValueError):
            strict_synthetic(x, "lexical")


class RealGoldManifestContract(unittest.TestCase):
    def test_annotated_manifest_shape(self):
        p, n = validate_manifest(make_manifest())
        self.assertEqual((len(p), len(n)), (20, 6))

    def test_reject_synthetic_audio(self):
        x = make_manifest()
        x["meetings"][1]["synthetic_audio"] = True
        with self.assertRaises(ValueError):
            validate_manifest(x)

    def test_reject_missing_human_review(self):
        x = make_manifest()
        x["cases"][3]["verified_from_audio"] = False
        with self.assertRaises(ValueError):
            validate_manifest(x)

    def test_reject_wrong_language_scope(self):
        x = make_manifest()
        x["cases"][3]["language"] = "pt"
        with self.assertRaises(ValueError):
            validate_manifest(x)

    def test_reject_cross_meeting_self_reference(self):
        x = make_manifest()
        x["cases"][20]["foreign_meeting_id"] = x["cases"][20]["meeting_id"]
        with self.assertRaises(ValueError):
            validate_manifest(x)

    def test_reject_inadequate_gold(self):
        x = make_manifest()
        x["cases"] = x["cases"][:-3]
        with self.assertRaises(ValueError):
            validate_manifest(x)


class RealAnswerScoringContract(unittest.TestCase):
    def setUp(self):
        self.case = make_manifest()["cases"][0]
        self.hit = {"source_kind": "meeting", "source_id": "m0",
                    "text": "[1] The verified fact was announced after review.",
                    "evidence": {"segment_ids": [1], "start_ms": 0, "end_ms": 200}}
        self.reply = {"status": "verbatim_citations_verified",
                      "sources": [self.hit],
                      "quotes": [{"source_number": 1, "quote": self.hit["text"]}]}

    def test_correct_source_and_audio_time(self):
        result = score_answerable(self.case, [self.hit], self.reply, True)
        self.assertTrue(result["answer_correct"])
        self.assertTrue(result["retrieval_recall_at_8"])

    def test_wrong_segment_is_rejected(self):
        reply = copy.deepcopy(self.reply)
        reply["sources"][0]["evidence"]["segment_ids"] = [99]
        self.assertFalse(score_answerable(self.case, [self.hit], reply, True)["answer_correct"])

    def test_wrong_meeting_is_leak(self):
        reply = copy.deepcopy(self.reply)
        reply["sources"][0]["source_id"] = "m1"
        result = score_answerable(self.case, [self.hit], reply, True)
        self.assertTrue(result["cross_scope_leak"])
        self.assertFalse(result["answer_correct"])

    def test_stale_revision_is_rejected(self):
        self.assertFalse(score_answerable(self.case, [self.hit], self.reply, False)["answer_correct"])

    def test_fabricated_quote_is_rejected(self):
        reply = copy.deepcopy(self.reply)
        reply["quotes"][0]["quote"] = "This was never in the source."
        self.assertFalse(score_answerable(self.case, [self.hit], reply, True)["answer_correct"])

    def test_negative_must_really_abstain(self):
        case = make_manifest()["cases"][21]
        self.assertTrue(score_negative(case, {"status": "abstained", "sources": [], "quotes": []})["abstention_correct"])
        self.assertFalse(score_negative(case, self.reply)["abstention_correct"])

    def test_real_summary_cannot_pass_all_untested(self):
        outcomes = ([{"kind": "answerable"} for _ in range(20)] +
                    [{"kind": "unanswerable"} for _ in range(6)])
        report = real_summary(outcomes, mode="lexical", timings=[1] * 26,
                              text_model="Saluki", embed_model=None)
        self.assertFalse(report["pass"])


class RealAudioWitnessContract(unittest.TestCase):
    def test_missing_transcribed_gold_is_not_accepted(self):
        from tools.meeting_knowledge_benchmark import source_gate
        case = make_manifest()["cases"][0]
        self.assertFalse(source_gate(case, [{"id": 1, "start_ms": 0,
                                            "end_ms": 200, "text": "incorrect ASR"}]))

    def test_cross_scope_requires_foreign_only_fact(self):
        from tools.meeting_knowledge_benchmark import negative_witness_gate
        case = make_manifest()["cases"][20]
        state = {"m0": (1, [{"id": 1, "text": "unrelated"}]),
                 "m1": (1, [{"id": 2, "text": "foreign verified fact"}]),
                 "m2": (1, [{"id": 3, "text": "other"}])}
        self.assertTrue(negative_witness_gate(case, state))
        state["m0"][1][0]["text"] = "foreign verified fact copied here"
        self.assertFalse(negative_witness_gate(case, state))


class ReleaseGateContract(unittest.TestCase):
    def test_synthetic_pass_never_unlocks_product(self):
        freeze = {"schema": FREEZE_SCHEMA, "pass": True,
                  "reports": {"lexical": {}, "hybrid": {}}}
        d = release_decision(freeze, expected_commit="a" * 40)
        self.assertTrue(d["local_synthetic_model_pass"])
        self.assertFalse(d["knowledge_release_pass"])
        self.assertFalse(d["overall_product_release_pass"])


if __name__ == "__main__":
    unittest.main()

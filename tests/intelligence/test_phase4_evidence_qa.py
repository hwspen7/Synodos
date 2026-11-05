"""Evidence QA cannot accept fabricated or non-attributable model output."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from intelligence.evidence_qa import evidence_prompt, verified_excerpt_answer
from intelligence.worker import process


HITS = [
    {"source_kind": "meeting", "source_id": "record-1", "revision": 5,
     "text": "[12] We approved the revised camera calibration on Monday.",
     "evidence": {"segment_ids": [12], "start_ms": 1000, "end_ms": 4300},
     "embedding": [0.3, 0.4]},
    {"source_kind": "meeting", "source_id": "record-2", "revision": 2,
     "text": "[9] The benchmark release has not yet been approved.",
     "evidence": {"segment_ids": [9], "start_ms": 2000, "end_ms": 4500},
     "embedding": None},
]


class EvidenceQATests(unittest.TestCase):
    def test_exact_sources_survive_and_unverified_model_prose_is_discarded(self):
        proposed = {
            "answer": "We are already in production. 100% success.",
            "quotes": [{
                "source_number": 1,
                "quote": "We approved the revised camera calibration on Monday.",
            }],
        }
        result = verified_excerpt_answer(proposed, HITS)
        self.assertEqual(result["status"], "verbatim_citations_verified")
        self.assertIn("We approved the revised camera", result["answer"])
        self.assertNotIn("100% success", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertNotIn("embedding", result["sources"][0])
        self.assertFalse(result["factual_entailment_verified"])
        self.assertTrue(result["review_required"])

    def test_fabricated_quote_or_nonexistent_citation_abstains(self):
        for proposed in [
            {"quotes": [{"source_number": 1, "quote": "Approved the whole release for production"}]},
            {"quotes": [{"source_number": 3, "quote": "We approved the revised camera calibration on Monday."}]},
            {"quotes": [{"source_number": True, "quote": "We approved the revised camera calibration on Monday."}]},
            {"quotes": [{"source_number": 1, "quote": "We approved"}]},
            {"answer": "Falsely asserted summary", "source_numbers": [1]},
        ]:
            with self.subTest(proposed=proposed):
                result = verified_excerpt_answer(proposed, HITS)
                self.assertEqual(result["status"], "abstained")
                self.assertEqual(result["sources"], [])
                self.assertNotIn("Falsely", result["answer"])

    def test_duplicate_and_disputed_quotes_do_not_pollute_evidence(self):
        first = {"source_number": 2, "quote": "The benchmark release has not yet been approved."}
        result = verified_excerpt_answer({"quotes": [
            first, first, {"source_number": 1, "quote": "That meeting was released"}
        ]}, HITS)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertIn("not yet been approved", result["answer"])

    def test_prompt_treats_transcript_as_untrusted_data(self):
        prompt = evidence_prompt("Was the release approved?", HITS)
        self.assertIn("untrusted", prompt)
        self.assertIn("Was the release approved?", prompt)
        self.assertIn("[12]", prompt)
        with self.assertRaises(ValueError):
            evidence_prompt("", HITS)

    def test_real_worker_path_preserves_quote_contract(self):
        class Gateway:
            def model_for(self, mode):
                return "local-test-model"
            def structured(self, *_args, **_kwargs):
                return {"answer": "Fictitious", "quotes": [{
                    "source_number": 2,
                    "quote": "The benchmark release has not yet been approved.",
                }]}
            def embed(self, *_args, **_kwargs):
                return None
        with patch("intelligence.worker.retrieval.search", return_value=HITS):
            result = process({
                "id": "job", "kind": "ask_evidence",
                "meeting_id": None,
                "payload": {"question": "Was release approved?"},
            }, Gateway())
        self.assertEqual(result["status"], "verbatim_citations_verified")
        self.assertNotIn("Fictitious", result["answer"])
        self.assertEqual(result["sources"][0]["revision"], 2)

    def test_evidence_endpoint_queues_distinct_opt_in_job(self):
        import asyncio
        from api.intelligence import AnswerRequest, ask_evidence
        with patch("api.intelligence._enabled"), patch(
            "api.intelligence.store.enqueue", return_value="job-123"
        ) as enqueue:
            result = asyncio.run(ask_evidence(
                AnswerRequest(question="What did we decide?")
            ))
        self.assertEqual(result["answer_policy"], "verbatim_source_quotes_only")
        self.assertEqual(result["job_id"], "job-123")
        enqueue.assert_called_once_with(
            "ask_evidence", None, {"question": "What did we decide?"}
        )

    def test_missing_source_fails_closed_without_model_inference(self):
        class Gateway:
            def structured(self, *_args, **_kwargs):
                raise AssertionError("No model request expected")
        with patch("intelligence.worker.retrieval.search", return_value=[]):
            result = process({
                "id": "job", "kind": "ask_evidence",
                "payload": {"question": "Any evidence?"},
            }, Gateway())
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()

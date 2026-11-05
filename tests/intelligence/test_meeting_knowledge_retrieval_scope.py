"""Meeting knowledge multilingual retrieval, scope and citation safety gates."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from intelligence import retrieval
from intelligence.evidence_qa import verified_excerpt_answer


class RetrievalQualityTests(unittest.TestCase):
    def test_full_corpus_recall_no_recent_250_limit(self):
        older = {"source_kind": "meeting", "source_id": "old-record",
                 "text": "The unique spectral calibration deadline is Friday.",
                 "embedding": None, "evidence": {"segment_ids": [9]},
                 "revision": 2}
        candidates = [
            {"source_kind": "meeting", "source_id": f"noise-{i}",
             "text": "Unrelated routine status meeting.", "embedding": None}
            for i in range(550)
        ] + [older]
        result = retrieval.rank("spectral calibration deadline", iter(candidates))
        self.assertEqual(result[0]["source_id"], "old-record")

    def test_unspaced_cjk_and_other_languages(self):
        corpus = [
            ("\u4e2d\u6587\u4f1a\u8bae\u5b89\u6392", "\u4f1a\u8bae\u5b89\u6392"),
            ("\u65e5\u672c\u8a9e\u4f1a\u8b70\u4e88\u5b9a", "\u4f1a\u8b70\u4e88\u5b9a"),
            ("\ud55c\uad6d\uc5b4\ud68c\uc758\uc77c\uc815", "\ud68c\uc758\uc77c\uc815"),
            ("La r\u00e9union confirme le budget.", "confirme budget"),
            ("El equipo confirma el calendario.", "equipo calendario"),
            ("Die Sitzung genehmigt den Plan.", "Sitzung Plan"),
            ("Reuni\u00e3o confirma o prazo.", "confirma prazo"),
            ("\u062a\u062d\u062f\u064a\u062b \u062e\u0637\u0629 \u0627\u0644\u0639\u0645\u0644",
             "\u062e\u0637\u0629 \u0627\u0644\u0639\u0645\u0644"),
            ("\u0938\u092e\u0940\u0915\u094d\u0937\u093e \u092a\u0930\u093f\u092f\u094b\u091c\u0928\u093e",
             "\u0938\u092e\u0940\u0915\u094d\u0937\u093e"),
            ("\u0995\u09cb\u09ac\u09be\u09b2\u09cd\u099f \u09af\u09be\u099a\u09be\u0987 \u0995\u09cb\u09a1",
             "\u0995\u09cb\u09ac\u09be\u09b2\u09cd\u099f \u09af\u09be\u099a\u09be\u0987"),

            ("\u041f\u0440\u043e\u0435\u043a\u0442 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d.",
             "\u041f\u0440\u043e\u0435\u043a\u0442"),
        ]
        for text, query in corpus:
            with self.subTest(query=query):
                self.assertGreater(retrieval.lexical_score(query, text), 0.0)

    def test_scope_validation_refuses_empty_or_malformed_lists(self):
        for ids in ([], ["../../secrets"], ["ok"] * 21, [42], "one"):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    retrieval.validate_meeting_ids(ids)
        self.assertEqual(
            retrieval.validate_meeting_ids(["valid_1", "valid_1", "valid_2"]),
            ["valid_1", "valid_2"],
        )

    def test_scope_and_embedding_are_forwarded_to_data_layer(self):
        class Gateway:
            cfg = SimpleNamespace(enable_embeddings=True)
            def embed(self, text):
                return [1.0, 0.0]
        candidate = {
            "source_kind": "meeting", "source_id": "chosen",
            "text": "No lexical keywords", "embedding": [1, 0],
        }
        with patch("intelligence.retrieval.store.iter_knowledge_candidates",
                   return_value=iter([candidate])) as stream:
            result = retrieval.search("semantic synonyms", Gateway(),
                                      meeting_ids=["chosen"])
        self.assertEqual(result[0]["source_id"], "chosen")
        stream.assert_called_once_with(
            meeting_ids=["chosen"], include_embeddings=True
        )

    def test_embedding_vectors_from_other_model_are_never_scored(self):
        candidates = [{
            "source_kind": "meeting", "source_id": "stale-model",
            "text": "Unrelated archived transcript",
            "embedding": [1, 0], "embedding_model": "other-embedder",
        }]
        bad = retrieval.rank(
            "nebular quintessence", candidates, [1, 0], 8,
            embedding_model="current-embedder",
        )
        self.assertEqual(bad, [])
        good = retrieval.rank(
            "nebular quintessence", [
                dict(candidates[0], embedding_model="current-embedder")
            ], [1, 0], 8, embedding_model="current-embedder",
        )
        self.assertEqual(len(good), 1)

    def test_query_with_no_evidence_never_promotes_model_prose(self):
        hits = [{"text": "Project Atlas has no shipping date.",
                 "source_kind": "meeting", "source_id": "m1",
                 "revision": 1, "evidence": {"segment_ids": [1]}}]
        response = verified_excerpt_answer({
            "answer": "The project ships tomorrow.",
            "quotes": [{"source_number": 1, "quote": "Shipping is tomorrow."}],
        }, hits)
        self.assertEqual(response["status"], "abstained")
        self.assertFalse(response["automatic_action_allowed"])


class ScopedEndpointTests(unittest.TestCase):
    def test_scoped_search_rejects_bad_identifier(self):
        from api.intelligence import SearchRequest, search
        with patch("api.intelligence._enabled") as enabled:
            enabled.return_value = SimpleNamespace(enable_embeddings=False)
            with self.assertRaises(HTTPException) as error:
                asyncio.run(search(SearchRequest(
                    query="Atlas", meeting_ids=["../private"]
                )))
        self.assertEqual(error.exception.status_code, 400)

    def test_completed_evidence_is_hidden_when_source_has_changed(self):
        from api.intelligence import get_job
        job = {
            "id": "test", "kind": "ask_evidence", "status": "completed",
            "result": {
                "answer": "Old private transcript",
                "status": "verbatim_citations_verified",
                "quotes": [{"quote": "Old private transcript"}],
                "sources": [{
                    "source_kind": "meeting", "source_id": "m1",
                    "revision": 3,
                }],
            },
        }
        with patch("api.intelligence._enabled"), patch(
            "api.intelligence.store.job_status", return_value=job,
        ), patch(
            "api.intelligence.store.evidence_revisions_current",
            return_value=False,
        ):
            response = asyncio.run(get_job(
                "230cabe7-516d-45ff-a88a-f01d4090a317"
            ))
        self.assertEqual(response["result"]["status"], "stale_source")
        self.assertEqual(response["result"]["sources"], [])
        self.assertNotIn("Old private transcript", response["result"]["answer"])

    def test_ask_scope_is_included_only_when_requested(self):
        from api.intelligence import AnswerRequest, ask_evidence
        with patch("api.intelligence._enabled"), patch(
            "api.intelligence.store.enqueue", return_value="test-job",
        ) as enqueue:
            asyncio.run(ask_evidence(AnswerRequest(
                question="What date?", meeting_ids=["meeting_1"],
            )))
        enqueue.assert_called_once_with(
            "ask_evidence", None,
            {"question": "What date?", "meeting_ids": ["meeting_1"]},
        )


if __name__ == "__main__":
    unittest.main()

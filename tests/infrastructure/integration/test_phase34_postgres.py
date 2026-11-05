"""Disposable PostgreSQL Phase 3/4 integration regression.

This suite uses actual PostgreSQL tables and revision locks, but no real
embedding or chat model. Live Ollama acceptance is a separate release gate.
"""
import os
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from intelligence import retrieval, store
from intelligence.worker import process
from sessions.manual_review import apply_review


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"),
                     "TEST_DATABASE_URL required for PostgreSQL integration")
class Phase34PostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        from storage.postgres_connection import PostgresDatabase
        from storage.postgres_meetings import PostgresMeetingStore

        cls.url = os.environ["TEST_DATABASE_URL"]
        with psycopg.connect(cls.url, connect_timeout=10) as conn:
            if conn.info.dbname != "meeting_test":
                raise RuntimeError("Refusing integration tests outside meeting_test")
        cls.db = PostgresDatabase(cls.url)
        cls.meetings = PostgresMeetingStore(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "DATABASE_URL": self.url,
            "MEETING_STORAGE_BACKEND": "postgres",
            "MEETING_AI_ENABLED": "1",
        })
        self.environment.start()
        self.created = []

    def tearDown(self):
        try:
            for mid in self.created:
                self.meetings.delete(mid)
        finally:
            self.environment.stop()

    def fixture(self, sentence, *, language="en"):
        mid = "test_phase34_" + uuid.uuid4().hex[:20]
        self.created.append(mid)
        self.meetings.save_state({
            "id": mid, "title": "Phase 3/4 isolated integration",
            "sample_rate": 16000,
            "segments": [{
                "id": 1, "start_sample": 0, "end_sample": 160000,
                "speaker_id": 1, "cluster": 0, "speaker_name": "Speaker 1",
                "text": sentence, "revision": 0, "language": language,
            }],
        })
        return mid

    @staticmethod
    def job(mid):
        return {"id": str(uuid.uuid4()), "meeting_id": mid,
                "kind": "index_meeting", "payload": {}}

    @staticmethod
    def gateway(*, vectors=False):
        class Model:
            cfg = SimpleNamespace(embed_model="fixture-vector-v1",
                                  enable_embeddings=vectors)

            def embed(self, text):
                if not vectors:
                    return None
                return [1.0, 0.0] if ("Vela" in text or "nebular" in text) else [0.0, 1.0]

            def model_for(self, _):
                return "fixture-only-model"

            def structured(self, _system, _prompt, **_kwargs):
                return {"quotes": [{"source_number": 1,
                                    "quote": "The Atlas demonstration is November 27, 2026."}]}

        return Model()

    def test_actual_postgres_scope_and_source_provenance(self):
        atlas = self.fixture("The Atlas demonstration is November 27, 2026.")
        other = self.fixture("The Atlas demonstration is November 30, 2026.")
        nonmatch = self.fixture("The navigation calibration is complete.")
        for mid in (atlas, other, nonmatch):
            result = process(self.job(mid), self.gateway())
            self.assertEqual(result["indexed_chunks"], 1)

        scoped = retrieval.search("Atlas demonstration", meeting_ids=[atlas])
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["source_id"], atlas)
        self.assertEqual(scoped[0]["evidence"]["segment_ids"], [1])
        self.assertEqual(scoped[0]["revision"], 1)

        qa = process({
            "id": "qa-test", "kind": "ask_evidence",
            "payload": {
                "question": "When is the Atlas demonstration?",
                "meeting_ids": [atlas],
            },
        }, self.gateway())
        self.assertEqual(qa["status"], "verbatim_citations_verified")
        self.assertEqual(qa["sources"][0]["source_id"], atlas)
        self.assertNotIn("November 30", qa["answer"])

        other_scoped = retrieval.search("Atlas", meeting_ids=[other])
        self.assertTrue(all(row["source_id"] == other for row in other_scoped))

    def test_manual_review_invalidates_stale_knowledge_and_retry(self):
        mid = self.fixture("The Atlas demonstration is November 20, 2026.")
        old_index = process(self.job(mid), self.gateway())
        original_hits = retrieval.search("Atlas demonstration", meeting_ids=[mid])
        self.assertEqual(len(original_hits), 1)

        snapshot = self.meetings.load_state(mid)
        version = snapshot["meta"]["_store_version"]
        snapshot.pop("meta", None)
        apply_review(
            snapshot, 1, 0,
            text="The Atlas demonstration is November 27, 2026.",
            reason="Synthetic manual revision",
        )
        self.meetings.save_state(snapshot, expected_store_revision=version)

        self.assertFalse(store.evidence_revisions_current(original_hits))
        self.assertEqual(retrieval.search("Atlas", meeting_ids=[mid]), [])
        with self.assertRaisesRegex(ValueError, "changed while indexing"):
            store.replace_chunks("meeting", mid, old_index["source_revision"], [{
                "text": "Old cached content",
                "evidence": {"segment_ids": [1]},
            }])

        refreshed = process(self.job(mid), self.gateway())
        self.assertGreater(refreshed["source_revision"], old_index["source_revision"])
        hits = retrieval.search("Atlas", meeting_ids=[mid])
        self.assertEqual(len(hits), 1)
        self.assertIn("November 27, 2026", hits[0]["text"])
        self.assertNotIn("November 20, 2026", hits[0]["text"])
        self.assertTrue(store.evidence_revisions_current(hits))

    def test_actual_jsonb_embeddings_enable_semantic_only_hit(self):
        mid = self.fixture("The Vela mission has a signed budget authorization.")
        process(self.job(mid), self.gateway(vectors=True))
        # Lexical query and source have deliberately no shared terms.
        bare = retrieval.search("nebular quintessence", meeting_ids=[mid])
        self.assertEqual(bare, [])
        semantic = retrieval.search("nebular quintessence",
                                    self.gateway(vectors=True),
                                    meeting_ids=[mid])
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0]["source_id"], mid)
        self.assertGreater(semantic[0]["score"], 0.5)

    def test_postgres_job_round_trip(self):
        mid = self.fixture("The Atlas demonstration is November 27, 2026.")
        job_id = store.enqueue("index_meeting", mid, {})
        claimed = store.claim()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], job_id)
        self.assertEqual(claimed["meeting_id"], mid)
        output = process(claimed, self.gateway())
        store.finish(job_id, output)
        result = store.job_status(job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["indexed_chunks"], 1)
        self.assertEqual(result["attempts"], 1)


if __name__ == "__main__":
    unittest.main()

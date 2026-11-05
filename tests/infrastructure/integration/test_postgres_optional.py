"""Optional live database round-trip on a disposable PostgreSQL database."""

import os
import uuid
import unittest


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"),
                     "Configure TEST_DATABASE_URL for disposable PostgreSQL tests")
class LivePostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from storage.postgres_connection import PostgresDatabase
        from storage.postgres_meetings import PostgresMeetingStore
        cls.db = PostgresDatabase(os.environ["TEST_DATABASE_URL"])
        cls.store = PostgresMeetingStore(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_create_save_export_and_delete(self):
        meeting_id = "test_" + uuid.uuid4().hex
        try:
            meta = self.store.create(meeting_id, "Test title")
            self.assertEqual(meta["id"], meeting_id)
            self.store.save_state({
                "id": meeting_id, "title": "Test title", "sample_rate": 16000,
                "segments": [{"id": 1, "start_sample": 0, "end_sample": 16000,
                              "cluster": -1, "speaker_id": 1, "text": "sample", "emb": None}],
            })
            self.assertEqual(self.store.list_segments(meeting_id)[0]["text"], "sample")
            self.assertIn("sample", self.store.export(meeting_id, "md"))
            self.assertEqual(self.store.load_state(meeting_id)["segments"][0]["text"], "sample")
        finally:
            self.store.delete(meeting_id)
        self.assertFalse(self.store.exists(meeting_id))

    def test_library_transactions_and_read_after_write(self):
        from storage.postgres_libraries import PostgresSpeakerStore, PostgresHotwordStore
        speaker_store = PostgresSpeakerStore(self.db)
        hotword_store = PostgresHotwordStore(self.db)
        speaker = speaker_store.create("Database test speaker")
        library = hotword_store.create("Database test hotwords", ["alpha", "beta"])
        try:
            self.assertEqual(speaker_store.get_for_matching([speaker["id"]])[0]["name"],
                             "Database test speaker")
            self.assertEqual(hotword_store.get_words_for_meeting([library["id"]]), ["alpha", "beta"])
        finally:
            speaker_store.delete_speaker(speaker["id"])
            hotword_store.delete(library["id"])

    def test_snapshot_compare_and_swap_preserves_latest_review(self):
        from sessions.manual_review import ReviewConflict
        mid = "test_" + uuid.uuid4().hex
        seg = {
            "id": 5, "start_sample": 0, "end_sample": 16000,
            "cluster": -1, "speaker_id": 1, "text": "initial",
            "revision": 0, "emb": None,
        }
        try:
            self.store.save_state({"id": mid, "segments": [seg]})
            first = self.store.load_state(mid)
            second = self.store.load_state(mid)
            version = first["meta"]["_store_version"]
            self.assertEqual(version, second["meta"]["_store_version"])
            second.pop("meta", None)
            second["segments"][0]["text"] = "confirmed newer revision"
            self.store.save_state(
                second, expected_store_revision=version,
            )
            first.pop("meta", None)
            first["segments"][0]["text"] = "stale correction"
            with self.assertRaises(ReviewConflict):
                self.store.save_state(
                    first, expected_store_revision=version,
                )
            self.assertEqual(
                self.store.load_state(mid)["segments"][0]["text"],
                "confirmed newer revision",
            )
            self.assertGreater(
                self.store.load_state(mid)["meta"]["_store_version"], version,
            )
        finally:
            self.store.delete(mid)

    def test_duplicate_segment_rolls_back_entire_save(self):
        mid = "test_" + uuid.uuid4().hex
        seg = {"id": 1, "start_sample": 0, "end_sample": 16000,
               "cluster": -1, "speaker_id": 1, "text": "safe", "emb": None}
        try:
            self.store.save_state({"id": mid, "segments": [seg]})
            with self.assertRaises(Exception):
                self.store.save_state({"id": mid, "segments": [dict(seg), dict(seg)]})
            self.assertEqual(self.store.list_segments(mid)[0]["text"], "safe")
            self.assertEqual(len(self.store.list_segments(mid)), 1)
        finally:
            self.store.delete(mid)

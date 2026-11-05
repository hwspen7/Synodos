"""Phase 3 meeting index must never replace newer transcript evidence."""
from __future__ import annotations

from contextlib import contextmanager
import unittest
from unittest.mock import patch

from intelligence import store


class FakeConnection:
    def __init__(self, revision):
        self.revision = revision
        self.commands = []

    def execute(self, sql, params=()):
        self.commands.append((sql, params))
        if "SELECT revision FROM meeting_records" in sql:
            return self

    def fetchone(self):
        return None if self.revision is None else (self.revision,)


class IndexCasTests(unittest.TestCase):
    def test_changed_meeting_refuses_to_delete_existing_index(self):
        fake = FakeConnection(revision=8)

        @contextmanager
        def connection():
            yield fake

        with patch.object(store, "connection", connection):
            with self.assertRaisesRegex(ValueError, "changed while indexing"):
                store.replace_chunks("meeting", "meeting-1", 7, [{
                    "text": "old", "evidence": {"segment_ids": [1]},
                }])
        self.assertEqual(len(fake.commands), 1)
        self.assertIn("FOR UPDATE", fake.commands[0][0])

    def test_deleted_meeting_cannot_be_indexed(self):
        fake = FakeConnection(revision=None)

        @contextmanager
        def connection():
            yield fake

        with patch.object(store, "connection", connection):
            with self.assertRaises(ValueError):
                store.replace_chunks("meeting", "deleted", 0, [])

    def test_current_meeting_can_atomically_replace_its_index(self):
        fake = FakeConnection(revision=9)

        @contextmanager
        def connection():
            yield fake

        with patch.object(store, "connection", connection), patch.object(
            store, "_json", side_effect=lambda x: x,
        ):
            store.replace_chunks("meeting", "meeting-1", 9, [{
                "text": "Human-corrected transcript",
                "evidence": {"segment_ids": [1]},
            }])
        self.assertEqual(len(fake.commands), 3)
        self.assertIn("FOR UPDATE", fake.commands[0][0])
        self.assertIn("DELETE FROM ai_knowledge", fake.commands[1][0])
        self.assertIn("INSERT INTO ai_knowledge", fake.commands[2][0])
        self.assertEqual(fake.commands[2][1][3], 9)

    def test_media_index_is_not_blocked_by_missing_meeting(self):
        fake = FakeConnection(revision=None)

        @contextmanager
        def connection():
            yield fake

        with patch.object(store, "connection", connection):
            store.replace_chunks("media", "media-1", 0, [])
        self.assertEqual(len(fake.commands), 1)
        self.assertIn("DELETE FROM ai_knowledge", fake.commands[0][0])

    def test_rejects_invalid_index_revision_and_batch(self):
        fake = FakeConnection(revision=1)

        @contextmanager
        def connection():
            yield fake

        with patch.object(store, "connection", connection):
            with self.assertRaises(ValueError):
                store.replace_chunks("meeting", "test", -1, [])
            with self.assertRaises(ValueError):
                store.replace_chunks("meeting", "test", 0, {})
        self.assertEqual(fake.commands, [])


if __name__ == "__main__":
    unittest.main()

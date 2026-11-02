"""A slow/disconnected client cannot create unbounded pending send tasks."""

from __future__ import annotations

import asyncio
import unittest

from sessions.outbox import enqueue_message


class OutboxTests(unittest.TestCase):
    def test_full_queue_drops_progress(self) -> None:
        q = asyncio.Queue(maxsize=2)
        enqueue_message(q, {"type": "transcript", "version": 1})
        enqueue_message(q, {"type": "transcript", "version": 2})
        self.assertFalse(enqueue_message(q, {"type": "state", "buffered_ms": 500}))
        self.assertEqual(q.qsize(), 2)

    def test_final_message_replaces_old_snapshot(self) -> None:
        q = asyncio.Queue(maxsize=2)
        enqueue_message(q, {"type": "transcript", "version": 1})
        enqueue_message(q, {"type": "transcript", "version": 2})
        self.assertTrue(enqueue_message(q, {"type": "closed", "saved": True}))
        self.assertEqual([q.get_nowait()["type"] for _ in range(2)], ["transcript", "closed"])

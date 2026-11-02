"""A new connection must wait for the previous processor to finish."""

from __future__ import annotations

import asyncio
import threading
import unittest

from api.session_lifecycle import meeting_connection_lock, stop_and_join


class FakeRecording:
    def __init__(self):
        self.stop_signal = threading.Event()
        self.finished = threading.Event()
        self.processor = threading.Thread(target=self.run, daemon=True)
        self.processor.start()

    def run(self):
        self.stop_signal.wait(1.0)
        self.finished.set()

    def stop(self):
        self.stop_signal.set()


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_thread_finalization(self):
        session = FakeRecording()
        self.assertTrue(await stop_and_join(session, 1.0))
        self.assertTrue(session.finished.is_set())
        self.assertFalse(session.processor.is_alive())

    async def test_same_meeting_connections_share_lock(self):
        first = meeting_connection_lock("test-meeting-lock")
        second = meeting_connection_lock("test-meeting-lock")
        self.assertIs(first, second)
        async with first:
            self.assertTrue(second.locked())

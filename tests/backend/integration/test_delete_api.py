"""Deleting an active meeting must stop its writer before removing the file."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.meetings import delete_meeting
from storage.meetings import MeetingStore


class DeleteApiTests(unittest.TestCase):
    def test_delete_stops_active_meeting_and_removes_registration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = MeetingStore(folder)
            mid = store.create(title="Temporary meeting")["id"]
            session = Mock()
            session.id = "active-session-id"
            session.ws.close = AsyncMock()
            active = {mid: session}
            by_id = {session.id: session}
            with (
                patch("api.meetings.meeting_store", store),
                patch("api.meetings.meeting_sessions", active),
                patch("api.meetings.sessions", by_id),
                patch("api.meetings.stop_and_join", new=AsyncMock(return_value=True)) as joined,
            ):
                result = asyncio.run(delete_meeting(mid))
            self.assertTrue(result["ok"])
            self.assertFalse(store.exists(mid))
            self.assertNotIn(mid, active)
            self.assertNotIn(session.id, by_id)
            joined.assert_awaited_once_with(session)

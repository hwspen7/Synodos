"""Do not acknowledge or export meeting state that failed to save."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from api.meetings import export_meeting, save_meeting
from storage.meetings import MeetingStore


class SaveApiTests(unittest.TestCase):
    def test_export_rejects_active_meeting_when_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = MeetingStore(folder)
            mid = store.create(title="Failure case")["id"]
            session = Mock()
            session.processor.request_save.return_value = False
            with patch("api.meetings.meeting_store", store), patch.dict(
                "api.meetings.meeting_sessions", {mid: session}
            ):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(export_meeting(mid, format="md"))
                self.assertEqual(captured.exception.status_code, 503)
                session.processor.request_save.assert_called_once()

    def test_save_endpoint_rejects_failed_write(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = MeetingStore(folder)
            mid = store.create(title="Failure case")["id"]
            session = Mock()
            session.processor.request_save.return_value = False
            with patch("api.meetings.meeting_store", store), patch.dict(
                "api.meetings.meeting_sessions", {mid: session}
            ):
                with self.assertRaises(HTTPException) as captured:
                    asyncio.run(save_meeting(mid))
                self.assertEqual(captured.exception.status_code, 500)

"""Process-wide model and stores, plus active sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from core.config import make_settings
from inference.models import ModelManager
from storage.meetings import MeetingStore
from storage.speakers import SpeakerStore
from storage.hotwords import HotwordStore

if TYPE_CHECKING:
    from sessions.session import Session

settings = make_settings()
models = ModelManager(settings)
meeting_store = MeetingStore(settings.meetings_dir)
speaker_store = SpeakerStore(settings.speakers_dir)
hotword_store = HotwordStore(settings.hotwords_dir)
sessions: dict[str, Session] = {}
meeting_sessions: dict[str, Session] = {}

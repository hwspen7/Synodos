"""Process-wide model and stores, plus active sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from core.config import make_settings
from inference.models import ModelManager
from storage.meetings import MeetingStore
from storage.speakers import SpeakerStore
from storage.hotwords import HotwordStore
from core.infrastructure_settings import load_infrastructure_settings
from core.validation import validate_application_settings

if TYPE_CHECKING:
    from sessions.session import Session

settings = make_settings()
validate_application_settings(settings)
infra = load_infrastructure_settings()
models = ModelManager(settings, infrastructure=infra)
database = None
if infra.database_backend == "postgres":
    from storage.postgres_connection import PostgresDatabase
    from storage.postgres_meetings import PostgresMeetingStore
    from storage.postgres_libraries import PostgresSpeakerStore, PostgresHotwordStore

    database = PostgresDatabase(infra.database_url, infra.database_pool_max)
    meeting_store = PostgresMeetingStore(database)
    speaker_store = PostgresSpeakerStore(database)
    hotword_store = PostgresHotwordStore(database)
else:
    meeting_store = MeetingStore(settings.meetings_dir)
    speaker_store = SpeakerStore(settings.speakers_dir)
    hotword_store = HotwordStore(settings.hotwords_dir)
sessions: dict[str, Session] = {}
meeting_sessions: dict[str, Session] = {}

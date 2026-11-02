"""Transactional PostgreSQL adapters for speaker and hotword libraries.

Library state is locked per kind and loaded on every operation, preventing
stale in-memory copies from overwriting another application's updates.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from psycopg.types.json import Jsonb

from storage.speakers import SpeakerStore
from storage.hotwords import HotwordStore
from storage.json_codec import json_safe


class _LibraryAdapter:
    def __init__(self, database, kind: str, key: str):
        self.database = database
        self._kind = kind
        self._key = key
        self._lock = threading.RLock()
        self._write_connection = None
        setattr(self, key, {})

    def _read_state(self, connection, lock: bool = False) -> None:
        query = "SELECT content FROM library_records WHERE kind = %s"
        if lock:
            query += " FOR UPDATE"
        row = connection.execute(query, (self._kind,)).fetchone()
        records = (row[0] if row else {}) or {}
        if self._kind == "speakers":
            for speaker in records.values():
                for vector in speaker.get("vectors", []):
                    vector["emb"] = np.asarray(vector["emb"], dtype=np.float32)
        setattr(self, self._key, records)

    def _save(self) -> None:
        if self._write_connection is None:
            raise RuntimeError("Database write attempted outside a library transaction")
        self._write_connection.execute(
            "UPDATE library_records SET content = %s, revision = revision + 1 WHERE kind = %s",
            (Jsonb(json_safe(getattr(self, self._key))), self._kind),
        )

    def _execute_read(self, fn: Callable, *args):
        with self._lock:
            with self.database.pool.connection() as conn:
                self._read_state(conn)
                return fn(self, *args)

    def _execute_write(self, fn: Callable, *args):
        with self._lock:
            with self.database.pool.connection() as conn:
                conn.execute(
                    "INSERT INTO library_records (kind, content) VALUES (%s, '{}'::jsonb) "
                    "ON CONFLICT (kind) DO NOTHING", (self._kind,),
                )
                self._read_state(conn, lock=True)
                self._write_connection = conn
                try:
                    return fn(self, *args)
                finally:
                    self._write_connection = None

    def import_records(self, records: dict) -> bool:
        """Only import when no row exists. Never silently replace live data."""
        with self._lock:
            with self.database.pool.connection() as conn:
                row = conn.execute(
                    "INSERT INTO library_records (kind, content) VALUES (%s, %s) "
                    "ON CONFLICT (kind) DO NOTHING RETURNING kind",
                    (self._kind, Jsonb(json_safe(records))),
                ).fetchone()
                return row is not None


class PostgresSpeakerStore(_LibraryAdapter, SpeakerStore):
    def __init__(self, database):
        _LibraryAdapter.__init__(self, database, "speakers", "_speakers")

    def list_speakers(self): return self._execute_read(SpeakerStore.list_speakers)
    def get_for_matching(self, ids): return self._execute_read(SpeakerStore.get_for_matching, ids)
    def create(self, name): return self._execute_write(SpeakerStore.create, name)
    def rename(self, sid, name): return self._execute_write(SpeakerStore.rename, sid, name)
    def add_vectors(self, sid, vectors): return self._execute_write(SpeakerStore.add_vectors, sid, vectors)
    def delete_vector(self, sid, vid): return self._execute_write(SpeakerStore.delete_vector, sid, vid)
    def delete_speaker(self, sid): return self._execute_write(SpeakerStore.delete_speaker, sid)


class PostgresHotwordStore(_LibraryAdapter, HotwordStore):
    def __init__(self, database):
        _LibraryAdapter.__init__(self, database, "hotwords", "_libraries")

    def list_hotwords(self): return self._execute_read(HotwordStore.list_hotwords)
    def get_words_for_meeting(self, ids, limit=200):
        return self._execute_read(HotwordStore.get_words_for_meeting, ids, limit)
    def get_summaries(self, ids): return self._execute_read(HotwordStore.get_summaries, ids)
    def create(self, name, words=None): return self._execute_write(HotwordStore.create, name, words)
    def update(self, hid, name=None, words=None):
        return self._execute_write(HotwordStore.update, hid, name, words)
    def delete(self, hid): return self._execute_write(HotwordStore.delete, hid)

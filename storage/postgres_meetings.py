"""PostgreSQL-backed meeting store with compatible API and atomic snapshots.

Meeting metadata and state are JSONB. Text segments are also indexed in a
relational table for queries. Embeddings remain arrays, not Python pickles.
"""

from __future__ import annotations

import re
import uuid
from storage.meetings import MeetingStore, _now_iso
from storage.json_codec import json_safe

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class PostgresMeetingStore(MeetingStore):
    def __init__(self, database):
        self.database = database
        # The legacy store's inherited helpers do not require a filesystem root.

    def valid_id(self, meeting_id: str) -> bool:
        return isinstance(meeting_id, str) and bool(_ID.fullmatch(meeting_id))

    def _check_id(self, meeting_id: str) -> None:
        if not self.valid_id(meeting_id):
            raise ValueError("invalid meeting id")

    def _read_payload(self, meeting_id: str) -> dict | None:
        self._check_id(meeting_id)
        with self.database.pool.connection() as conn:
            row = conn.execute(
                "SELECT meta, state, revision FROM meeting_records WHERE id = %s",
                (meeting_id,),
            ).fetchone()
        return {
            "meta": {**(row[0] or {}), "_store_version": int(row[2])},
            "state": row[1],
        } if row else None

    def exists(self, meeting_id: str) -> bool:
        self._check_id(meeting_id)
        with self.database.pool.connection() as conn:
            return conn.execute("SELECT 1 FROM meeting_records WHERE id = %s", (meeting_id,)).fetchone() is not None

    def create(self, meeting_id: str | None = None, title: str = "") -> dict:
        from psycopg.types.json import Jsonb
        mid = meeting_id or uuid.uuid4().hex
        self._check_id(mid)
        now = _now_iso()
        meta = {
            "id": mid, "title": (title or "Untitled Meeting").strip() or "Untitled Meeting",
            "created_at": now, "updated_at": now, "sample_rate": 16000,
            "merge_gap_s": 2.0, "segments": 0, "duration_ms": 0, "speakers": 0,
        }
        with self.database.pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO meeting_records(id, meta, state) VALUES (%s, %s, NULL) "
                "ON CONFLICT (id) DO NOTHING RETURNING meta", (mid, Jsonb(meta)),
            ).fetchone()
            if not row:
                row = conn.execute("SELECT meta FROM meeting_records WHERE id = %s", (mid,)).fetchone()
        return row[0]

    def list_meetings(self) -> list[dict]:
        with self.database.pool.connection() as conn:
            rows = conn.execute("SELECT meta FROM meeting_records ORDER BY updated_at DESC, id").fetchall()
        return [r[0] for r in rows]

    def save_state(self, state: dict, *,
                   expected_store_revision: int | None = None) -> dict:
        from psycopg.types.json import Jsonb
        mid = state["id"]
        self._check_id(mid)
        state = json_safe(state)
        sr = int(state.get("sample_rate", 16000))
        if sr <= 0:
            raise ValueError("Invalid sample rate")
        segs = state.get("segments") or []
        duration_ms = int(max((int(s["end_sample"]) for s in segs), default=0) * 1000 / sr)
        mapping = state.get("speaker_map") or {}
        stable = {int(s.get("speaker_id", mapping.get(str(s["cluster"]), s["cluster"]))) for s in segs}
        now = _now_iso()
        with self.database.pool.connection() as conn:
            # Serialize writes for this meeting in the same database transaction.
            conn.execute(
                "INSERT INTO meeting_records (id, meta, state) VALUES (%s, '{}'::jsonb, NULL) "
                "ON CONFLICT (id) DO NOTHING", (mid,),
            )
            from sessions.manual_review import ReviewConflict
            locked = conn.execute(
                "SELECT meta, revision FROM meeting_records WHERE id = %s FOR UPDATE",
                (mid,),
            ).fetchone()
            old, version = locked[0], int(locked[1])
            if (expected_store_revision is not None
                    and version != expected_store_revision):
                raise ReviewConflict(
                    f"Stale meeting snapshot {expected_store_revision}; current is {version}"
                )
            meta = {
                "id": mid,
                "title": (state.get("title") or "Untitled Meeting").strip() or "Untitled Meeting",
                "created_at": old.get("created_at") or now, "updated_at": now,
                "sample_rate": sr, "merge_gap_s": float(state.get("merge_gap_s", 2.0)),
                "segments": len(segs), "duration_ms": duration_ms, "speakers": len(stable),
            }
            conn.execute(
                "UPDATE meeting_records SET meta = %s, state = %s, updated_at = now(), "
                "revision = revision + 1 WHERE id = %s", (Jsonb(meta), Jsonb(state), mid),
            )
            conn.execute("DELETE FROM meeting_segments WHERE meeting_id = %s", (mid,))
            for seg in segs:
                conn.execute(
                    "INSERT INTO meeting_segments (meeting_id, segment_id, start_sample, end_sample, speaker_id, content) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (mid, int(seg["id"]), int(seg["start_sample"]), int(seg["end_sample"]),
                     int(seg.get("speaker_id", seg.get("cluster", -1))), str(seg.get("text") or "")),
                )
        return meta

    def delete(self, meeting_id: str) -> bool:
        self._check_id(meeting_id)
        with self.database.pool.connection() as conn:
            # Remove meeting-linked AI state in one transaction.
            # Shared uploaded media is not meeting-owned.
            conn.execute("DELETE FROM ai_knowledge WHERE source_kind='meeting' AND source_id=%s", (meeting_id,))
            conn.execute("DELETE FROM ai_translations WHERE meeting_id=%s", (meeting_id,))
            conn.execute("DELETE FROM ai_artifacts WHERE meeting_id=%s", (meeting_id,))
            conn.execute("DELETE FROM ai_jobs WHERE meeting_id=%s", (meeting_id,))
            deleted = conn.execute("DELETE FROM meeting_records WHERE id = %s RETURNING id", (meeting_id,)).fetchone()
        return deleted is not None

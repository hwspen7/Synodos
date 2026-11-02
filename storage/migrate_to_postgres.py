"""Explicit, repeatable migration from trusted local pickle data to PostgreSQL.

Run with --apply after inspecting --dry-run output and backing up the files.
Never import pickle from untrusted sources.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from storage.meetings import MeetingStore


def migrate(dsn: str, meetings_dir: Path, speakers_dir: Path, hotwords_dir: Path,
            apply: bool = False) -> dict:
    source = MeetingStore(meetings_dir)
    source_meetings = source.list_meetings()
    result = {"meetings_found": len(source_meetings), "meetings_imported": 0,
              "meetings_skipped": 0, "speaker_library_imported": False,
              "hotword_library_imported": False, "mode": "apply" if apply else "dry-run"}
    if not apply:
        return result
    from storage.postgres_connection import PostgresDatabase
    from storage.postgres_meetings import PostgresMeetingStore
    from storage.postgres_libraries import PostgresSpeakerStore, PostgresHotwordStore
    from psycopg.types.json import Jsonb

    db = PostgresDatabase(dsn)
    try:
        target = PostgresMeetingStore(db)
        for item in source_meetings:
            mid = item["id"]
            if target.exists(mid):
                result["meetings_skipped"] += 1
                continue
            state = source.load_state(mid)
            if state and any(key != "meta" for key in state):
                state["id"] = mid
                target.save_state(state)
            else:
                target.create(mid, item.get("title") or "")
            # Preserve original meeting timestamps and other display metadata.
            created_meta = target.get_meta(mid) or {}
            created_meta.update({key: item[key] for key in
                                 ("created_at", "updated_at") if item.get(key)})
            with db.pool.connection() as conn:
                conn.execute(
                    "UPDATE meeting_records SET meta = %s, updated_at = %s::timestamptz WHERE id = %s",
                    (Jsonb(created_meta), created_meta.get("updated_at") or created_meta["created_at"], mid),
                )
            result["meetings_imported"] += 1
        for path, name, store in (
            (speakers_dir / "speakers.pkl", "speakers", PostgresSpeakerStore(db)),
            (hotwords_dir / "hotwords.pkl", "libraries", PostgresHotwordStore(db)),
        ):
            if path.exists():
                # Explicit local-only trusted migration; not a remote upload endpoint.
                payload = pickle.loads(path.read_bytes())
                imported = store.import_records(payload.get(name, {}) or {})
                result["speaker_library_imported" if name == "speakers" else "hotword_library_imported"] = imported
    finally:
        db.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meetings-dir", type=Path, default=Path("meetings"))
    parser.add_argument("--speakers-dir", type=Path, default=Path("speakers"))
    parser.add_argument("--hotwords-dir", type=Path, default=Path("hotwords"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dsn = os.environ.get("DATABASE_URL", "")
    if args.apply and not dsn:
        parser.error("DATABASE_URL must be configured before --apply")
    result = migrate(dsn, args.meetings_dir, args.speakers_dir, args.hotwords_dir, args.apply)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

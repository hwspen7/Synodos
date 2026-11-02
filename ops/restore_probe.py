"""Validate a PostgreSQL custom backup via a dedicated disposable database."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from ops.backup_manager import BackupError, run_pg, verify_archive

VERIFY_DATABASE = "meeting_restore_verify"
ALLOWED_TABLES = {"meeting_records", "meeting_segments", "library_records", "meeting_schema_migrations"}


def validate_target(database: str) -> None:
    if database != VERIFY_DATABASE:
        raise BackupError("Restore verification is restricted to meeting_restore_verify")


def restore_into_scratch(archive: Path, pg_bin: Path, database: str) -> dict:
    validate_target(database)
    pg_restore = pg_bin / "pg_restore.exe"
    psql = pg_bin / "psql.exe"
    verify_archive(archive, archive.with_suffix(".json"), pg_restore)
    common = ["--no-password", "--host=127.0.0.1", "--port=5432", "--username=meeting", f"--dbname={database}"]
    # A dedicated verification database must be provisioned by an administrator.
    # Reject databases containing any unrelated user tables before a --clean restore.
    query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    tables = run_pg(psql, [*common, "--tuples-only", "--no-align", "--command=" + query], timeout=30)
    found = {line.strip() for line in tables.stdout.splitlines() if line.strip()}
    unexpected = found - ALLOWED_TABLES
    if unexpected:
        raise BackupError("Verification database contains unexpected tables; refusing to overwrite")
    run_pg(pg_restore, [
        "--no-password", "--host=127.0.0.1", "--port=5432", "--username=meeting",
        f"--dbname={database}", "--clean", "--if-exists", "--no-owner", "--no-acl",
        "--exit-on-error", str(archive),
    ])
    counts = run_pg(psql, [*common, "--tuples-only", "--no-align", "--command=" +
        "SELECT (SELECT count(*) FROM meeting_records), "
        "(SELECT count(*) FROM meeting_segments), (SELECT count(*) FROM library_records)"], timeout=30)
    result = counts.stdout.strip().split("|")
    if len(result) != 3:
        raise BackupError("Unexpected restore verification query output")
    return {"restore_database": database, "meeting_records": int(result[0]),
            "meeting_segments": int(result[1]), "library_records": int(result[2]),
            "status": "full_restore_passed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pg-bin", type=Path, default=Path(r"C:\Program Files\PostgreSQL\17\bin"))
    parser.add_argument("--database", default=VERIFY_DATABASE)
    parser.add_argument("--confirm-disposable-database", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_disposable_database:
        print("Use --confirm-disposable-database after reserving meeting_restore_verify for destructive restore checks", file=sys.stderr)
        return 2
    try:
        result = restore_into_scratch(args.archive, args.pg_bin, args.database)
    except (BackupError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"RESTORE CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print("FULL RESTORE VERIFICATION PASSED")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

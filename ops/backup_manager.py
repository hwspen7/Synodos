"""Verified logical PostgreSQL backups and conservative retention.

Only PostgreSQL client programs handle database data. This module does not
connect to, mutate, or inspect the application's running Python objects.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

NAME = re.compile(r"^meeting_\d{8}T\d{6}Z_[0-9a-f]{8}\.dump$")
MANIFEST_VERSION = 1


class BackupError(RuntimeError):
    """A backup or archive validation step failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pg(program: Path, arguments: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    if not program.is_file():
        raise BackupError(f"PostgreSQL client not found: {program}")
    env = dict(os.environ)
    env["PGCONNECT_TIMEOUT"] = "5"
    env["PGCLIENTENCODING"] = "UTF8"
    # Never prompt in scheduled backups. Credentials are supplied by the wrapper.
    result = subprocess.run(
        [str(program), *arguments], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        # PostgreSQL messages can contain connection details. Avoid echoing stderr.
        raise BackupError(f"{program.name} failed (exit {result.returncode}); see PostgreSQL service logs")
    return result


def verify_archive(archive: Path, manifest: Path, pg_restore: Path) -> dict:
    if not archive.is_file() or archive.is_symlink():
        raise BackupError("Backup archive is missing or is a symbolic link")
    if not manifest.is_file() or manifest.is_symlink():
        raise BackupError("Backup manifest is missing or is a symbolic link")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise BackupError("Cannot parse backup manifest") from exc
    if data.get("format_version") != MANIFEST_VERSION or data.get("archive") != archive.name:
        raise BackupError("Invalid archive manifest metadata")
    if data.get("size_bytes") != archive.stat().st_size or data.get("size_bytes", 0) <= 0:
        raise BackupError("Archive size mismatch")
    if data.get("sha256") != sha256_file(archive):
        raise BackupError("Archive checksum mismatch")
    run_pg(pg_restore, ["--list", str(archive)], timeout=120)
    return data


def safe_entries(root: Path, pg_restore: Path) -> list[tuple[Path, dict]]:
    """Return only backups that are valid and owned by this tool's naming scheme."""
    entries: list[tuple[Path, dict]] = []
    if not root.is_dir():
        return entries
    for archive in root.glob("meeting_*.dump"):
        if not NAME.fullmatch(archive.name):
            continue
        manifest = archive.with_suffix(".json")
        try:
            data = verify_archive(archive, manifest, pg_restore)
            stamp = datetime.fromisoformat(data["created_at_utc"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                continue
        except (BackupError, KeyError, ValueError):
            continue
        entries.append((archive, data))
    entries.sort(key=lambda item: item[1]["created_at_utc"], reverse=True)
    return entries


def prune(root: Path, pg_restore: Path, keep_days: int, keep_latest: int, now: datetime | None = None) -> list[str]:
    if keep_days < 1 or keep_latest < 1:
        raise ValueError("Retention settings must be positive")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    removed: list[str] = []
    for index, (archive, data) in enumerate(safe_entries(root, pg_restore)):
        if index < keep_latest:
            continue
        created = datetime.fromisoformat(data["created_at_utc"].replace("Z", "+00:00"))
        if created >= cutoff:
            continue
        # Retention only deletes validated files created by this backup tool.
        manifest = archive.with_suffix(".json")
        if archive.is_symlink() or manifest.is_symlink():
            continue
        archive.unlink()
        manifest.unlink()
        removed.append(archive.name)
    return removed


def make_backup(
    root: Path, pg_dump: Path, pg_restore: Path, database: str = "meeting",
    host: str = "127.0.0.1", port: int = 5432, user: str = "meeting",
    retention_days: int = 30, keep_latest: int = 7,
) -> dict:
    if database != "meeting":
        raise BackupError("Only the formal meeting database can be backed up with this command")
    if not host in {"127.0.0.1", "localhost"} or user != "meeting" or not 1 <= port <= 65535:
        raise BackupError("Unexpected database destination or user")
    if not (1 <= retention_days <= 3650 and 1 <= keep_latest <= 365):
        raise BackupError("Retention values out of range")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise BackupError("Refusing to write into a symbolic-link backup directory")
    now = datetime.now(timezone.utc)
    name = f"meeting_{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.dump"
    archive = root / name
    partial = root / (name + ".partial")
    manifest = archive.with_suffix(".json")
    partial_manifest = manifest.with_name(manifest.name + ".partial")
    try:
        run_pg(pg_dump, [
            "--format=custom", "--no-owner", "--no-acl", "--no-password",
            f"--host={host}", f"--port={port}", f"--username={user}",
            f"--dbname={database}", f"--file={partial}",
        ])
        if not partial.is_file() or partial.stat().st_size == 0:
            raise BackupError("pg_dump returned without producing a nonempty archive")
        # pg_restore's catalog read catches invalid archive headers; checksum catches later bit-rot.
        run_pg(pg_restore, ["--list", str(partial)], timeout=120)
        details = {
            "format_version": MANIFEST_VERSION,
            "archive": archive.name,
            "created_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "database": database,
            "size_bytes": partial.stat().st_size,
            "sha256": sha256_file(partial),
            "validation": "sha256_and_pg_restore_catalog",
        }
        partial_manifest.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, archive)
        os.replace(partial_manifest, manifest)
        verify_archive(archive, manifest, pg_restore)
        removed = prune(root, pg_restore, retention_days, keep_latest, now)
        return {"archive": str(archive), "size_bytes": details["size_bytes"],
                "sha256": details["sha256"], "retention_deleted": removed}
    finally:
        partial.unlink(missing_ok=True)
        partial_manifest.unlink(missing_ok=True)


def clients(bin_path: Path) -> tuple[Path, Path]:
    return bin_path / "pg_dump.exe", bin_path / "pg_restore.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or check verified PostgreSQL backups")
    parser.add_argument("operation", choices=["backup", "verify", "list"])
    parser.add_argument("--pg-bin", type=Path, default=Path(r"C:\Program Files\PostgreSQL\17\bin"))
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--keep-latest", type=int, default=7)
    args = parser.parse_args(argv)
    pg_dump, pg_restore = clients(args.pg_bin)
    try:
        if args.operation == "backup":
            data = make_backup(args.backup_root, pg_dump, pg_restore,
                               retention_days=args.retention_days, keep_latest=args.keep_latest)
        elif args.operation == "verify":
            if args.archive is None:
                parser.error("--archive is required for verify")
            data = verify_archive(args.archive, args.archive.with_suffix(".json"), pg_restore)
        else:
            data = [{"archive": str(path), "created_at_utc": info["created_at_utc"]}
                    for path, info in safe_entries(args.backup_root, pg_restore)]
        print(json.dumps(data, indent=2))
        return 0
    except (BackupError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"BACKUP OPERATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

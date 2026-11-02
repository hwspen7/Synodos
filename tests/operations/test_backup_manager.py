"""Deterministic tests of backup integrity, cleanup, and failure handling."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ops.backup_manager import BackupError, make_backup, prune, sha256_file, verify_archive


class BackupManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dump_bin = self.root / "pg_dump.exe"
        self.restore_bin = self.root / "pg_restore.exe"
        self.dump_bin.write_bytes(b"binary")
        self.restore_bin.write_bytes(b"binary")

    def _fake_pg(self, program, args, timeout=600):
        if program.name == "pg_dump.exe":
            dest = next(a.split("=", 1)[1] for a in args if a.startswith("--file="))
            Path(dest).write_bytes(b"PGDMP fake test archive content")
        elif program.name == "pg_restore.exe":
            if args[0] != "--list":
                raise AssertionError(args)
        else:
            raise AssertionError(program)
        return None

    @patch("ops.backup_manager.run_pg")
    def test_creates_verifiable_backup(self, run):
        run.side_effect = self._fake_pg
        result = make_backup(self.root / "backups", self.dump_bin, self.restore_bin)
        archive = Path(result["archive"])
        self.assertTrue(archive.is_file())
        self.assertEqual(verify_archive(archive, archive.with_suffix(".json"), self.restore_bin)["sha256"],
                         sha256_file(archive))
        self.assertFalse(any(x.name.endswith(".partial") for x in archive.parent.iterdir()))

    @patch("ops.backup_manager.run_pg")
    def test_corruption_is_detected(self, run):
        run.side_effect = self._fake_pg
        result = make_backup(self.root / "backups", self.dump_bin, self.restore_bin)
        archive = Path(result["archive"])
        archive.write_bytes(archive.read_bytes() + b"tamper")
        with self.assertRaises(BackupError):
            verify_archive(archive, archive.with_suffix(".json"), self.restore_bin)

    @patch("ops.backup_manager.run_pg")
    def test_failed_backup_does_not_leave_partial_artifact(self, run):
        run.side_effect = BackupError("mock failure")
        with self.assertRaises(BackupError):
            make_backup(self.root / "backups", self.dump_bin, self.restore_bin)
        self.assertEqual(list((self.root / "backups").iterdir()), [])

    @patch("ops.backup_manager.run_pg")
    def test_rejects_unexpected_database(self, run):
        with self.assertRaises(BackupError):
            make_backup(self.root / "backups", self.dump_bin, self.restore_bin, database="meeting_test")
        run.assert_not_called()

    @patch("ops.backup_manager.run_pg")
    def test_prune_keeps_seven_latest_and_unrelated_files(self, run):
        run.side_effect = self._fake_pg
        root = self.root / "backups"
        root.mkdir()
        old = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(9):
            stamp = (old + timedelta(seconds=i)).isoformat(timespec="seconds").replace("+00:00", "Z")
            name = f"meeting_20260101T0000{i:02d}Z_{i:08x}.dump"
            archive = root / name
            archive.write_bytes(b"PGDMP test")
            archive.with_suffix(".json").write_text(json.dumps({
                "format_version": 1, "archive": name, "created_at_utc": stamp,
                "size_bytes": archive.stat().st_size, "sha256": sha256_file(archive),
            }), encoding="utf-8")
        unexpected = root / "personal_notes.dump"
        unexpected.write_bytes(b"Never delete this")
        removed = prune(root, self.restore_bin, keep_days=30, keep_latest=7)
        self.assertEqual(len(removed), 2)
        self.assertEqual(len(list(root.glob("meeting_*.dump"))), 7)
        self.assertTrue(unexpected.exists())

    @patch("ops.backup_manager.run_pg")
    def test_retention_skips_corrupt_archive(self, run):
        run.side_effect = self._fake_pg
        root = self.root / "backups"
        root.mkdir()
        name = "meeting_20260101T000000Z_deadbeef.dump"
        archive = root / name
        archive.write_bytes(b"corrupted")
        archive.with_suffix(".json").write_text(json.dumps({
            "format_version": 1, "archive": name,
            "created_at_utc": "2026-01-01T00:00:00Z", "size_bytes": 9,
            "sha256": "invalid",
        }), encoding="utf-8")
        self.assertEqual(prune(root, self.restore_bin, 1, 1), [])
        self.assertTrue(archive.exists())


if __name__ == "__main__":
    unittest.main()

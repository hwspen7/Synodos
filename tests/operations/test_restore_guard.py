"""Restore must refuse production and unknown database tables."""

from pathlib import Path
import unittest
from unittest.mock import patch

from ops.backup_manager import BackupError
from ops.restore_probe import restore_into_scratch, validate_target


class RestoreGuardTests(unittest.TestCase):
    def test_production_target_is_refused(self):
        with self.assertRaises(BackupError):
            validate_target("meeting")

    def test_test_database_is_not_restore_target(self):
        with self.assertRaises(BackupError):
            validate_target("meeting_test")

    def test_disposable_target_only(self):
        validate_target("meeting_restore_verify")

    @patch("ops.restore_probe.verify_archive")
    @patch("ops.restore_probe.run_pg")
    def test_unrecognized_table_prevents_restore(self, run, verify):
        class Output:
            stdout = "sensitive_other_project\n"
        run.return_value = Output()
        with self.assertRaises(BackupError):
            restore_into_scratch(Path("sample.dump"), Path("bin"), "meeting_restore_verify")
        self.assertEqual(run.call_count, 1)

    @patch("ops.restore_probe.verify_archive")
    @patch("ops.restore_probe.run_pg")
    def test_real_restore_verification_queries_counts(self, run, verify):
        class Output:
            def __init__(self, stdout):
                self.stdout = stdout
        run.side_effect = [Output("meeting_records\nmeeting_segments\n"), Output(""), Output("4|8|2\n")]
        result = restore_into_scratch(Path("sample.dump"), Path("bin"), "meeting_restore_verify")
        self.assertEqual(result["meeting_records"], 4)
        self.assertEqual(result["meeting_segments"], 8)
        self.assertEqual(result["library_records"], 2)
        self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()

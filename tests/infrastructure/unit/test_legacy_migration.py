import tempfile
import unittest
from pathlib import Path
from storage.meetings import MeetingStore
from storage.migrate_to_postgres import migrate


class MigrationPreviewTests(unittest.TestCase):
    def test_dry_run_never_opens_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MeetingStore(root / "meetings")
            store.create("demo123", "Local record")
            result = migrate("", root / "meetings", root / "speakers", root / "hotwords", apply=False)
            self.assertEqual(result["meetings_found"], 1)
            self.assertEqual(result["meetings_imported"], 0)
            self.assertEqual(result["mode"], "dry-run")

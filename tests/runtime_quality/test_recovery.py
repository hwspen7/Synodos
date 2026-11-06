"""Phase 07E explicit save interval validation."""
import unittest
from unittest.mock import patch
from runtime_quality.recovery import autosave_seconds


class AutosaveTests(unittest.TestCase):
    def test_default_enabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(autosave_seconds(), 60)

    def test_explicit_disable(self):
        with patch.dict("os.environ", {"MEETING_PHASE07_AUTOSAVE_SECONDS": "0"}):
            self.assertEqual(autosave_seconds(), 0)

    def test_valid_minimum(self):
        with patch.dict("os.environ", {"MEETING_PHASE07_AUTOSAVE_SECONDS": "15"}):
            self.assertEqual(autosave_seconds(), 15)

    def test_invalid_short_interval(self):
        with patch.dict("os.environ", {"MEETING_PHASE07_AUTOSAVE_SECONDS": "2"}):
            with self.assertRaises(ValueError):
                autosave_seconds()

    def test_invalid_non_numeric(self):
        with patch.dict("os.environ", {"MEETING_PHASE07_AUTOSAVE_SECONDS": "invalid"}):
            with self.assertRaises(ValueError):
                autosave_seconds()

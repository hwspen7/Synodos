"""Regression coverage for local intelligence paths without a user profile."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from intelligence.config import load_config


class IntelligenceMediaRootTests(unittest.TestCase):
    def test_explicit_media_path_never_asks_for_home_directory(self):
        with TemporaryDirectory() as directory:
            expected = Path(directory).resolve()
            with patch.dict(os.environ, {
                'MEETING_AI_MEDIA_ROOT': str(expected),
            }, clear=True), patch(
                'intelligence.config.Path.home',
                side_effect=RuntimeError('Could not determine home directory.'),
            ):
                self.assertEqual(load_config().media_root, expected)

    def test_missing_home_uses_repository_local_fallback(self):
        expected = Path(__file__).resolve().parents[2] / 'MeetingMedia'
        with patch.dict(os.environ, {}, clear=True), patch(
            'intelligence.config.Path.home',
            side_effect=RuntimeError('Could not determine home directory.'),
        ):
            self.assertEqual(load_config().media_root, expected.resolve())

    def test_available_home_preserves_existing_default(self):
        with TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            with patch.dict(os.environ, {}, clear=True), patch(
                'intelligence.config.Path.home', return_value=home,
            ):
                self.assertEqual(
                    load_config().media_root, home / 'MeetingMedia'
                )

    def test_explicit_unresolvable_home_reports_config_error(self):
        with patch.dict(os.environ, {
            'MEETING_AI_MEDIA_ROOT': '~/MeetingMedia',
        }, clear=True), patch(
            'intelligence.config.Path.expanduser',
            side_effect=RuntimeError('Could not determine home directory.'),
        ):
            with self.assertRaisesRegex(ValueError, 'MEETING_AI_MEDIA_ROOT'):
                load_config()


if __name__ == '__main__':
    unittest.main()

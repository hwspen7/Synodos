"""Validate configuration before accepting requests or opening databases."""

import unittest
from foundation.settings import load_infrastructure_settings


class SettingsTests(unittest.TestCase):
    def test_default_is_safe_and_legacy(self):
        settings = load_infrastructure_settings({})
        self.assertEqual(settings.database_backend, "legacy")
        self.assertFalse(settings.allow_nonbrowser_websocket)
        self.assertNotIn("database_url", settings.safe_summary())

    def test_postgres_requires_dsn(self):
        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            load_infrastructure_settings({"MEETING_STORAGE_BACKEND": "postgres"})

    def test_bad_origin_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            load_infrastructure_settings({"APP_ALLOWED_ORIGINS": "*"})

    def test_bad_queue_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            load_infrastructure_settings({"INFERENCE_QUEUE_SIZE": "0"})

    def test_invalid_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            load_infrastructure_settings({"APP_ENABLE_METRICS": "sometimes"})

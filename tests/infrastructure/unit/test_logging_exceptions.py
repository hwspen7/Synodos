"""Check that structured error logs include enough evidence to diagnose GPU failures."""

from __future__ import annotations

import json
import logging
import unittest

from core.logging_setup import JsonFormatter


class JsonLoggingTracebackTests(unittest.TestCase):
    def test_exception_type_message_and_traceback_are_preserved(self):
        try:
            raise RuntimeError("Could not locate cudnn_ops64_9.dll")
        except RuntimeError:
            import sys
            failure = sys.exc_info()

        record = logging.LogRecord(
            "sessions.processor", logging.ERROR, __file__, 40,
            "session test cycle failed", (), failure,
        )
        result = json.loads(JsonFormatter().format(record))
        self.assertEqual(result["exception_type"], "RuntimeError")
        self.assertIn("cudnn_ops64_9.dll", result["exception_message"])
        self.assertIn("RuntimeError:", result["exception_traceback"])
        self.assertIn("test_exception_type_message_and_traceback_are_preserved", result["exception_traceback"])

    def test_database_url_redacted_inside_traceback(self):
        try:
            raise RuntimeError("failed at postgresql://meeting:secret@127.0.0.1:5432/meeting")
        except RuntimeError:
            import sys
            failure = sys.exc_info()

        record = logging.LogRecord(
            "sessions.processor", logging.ERROR, __file__, 60,
            "connection failed", (), failure,
        )
        result = JsonFormatter().format(record)
        self.assertNotIn("secret", result)
        self.assertIn("postgresql://[REDACTED]", result)


if __name__ == "__main__":
    unittest.main()

import json
import logging
import unittest
from core.metrics import Metrics
from core.logging_setup import JsonFormatter


class DiagnosticsTests(unittest.TestCase):
    def test_metrics_counts_and_gauges(self):
        m = Metrics()
        m.increment("saved")
        m.increment("saved", 2)
        m.gauge("queued", 3)
        self.assertEqual(m.snapshot(), {"counters": {"saved": 3}, "gauges": {"queued": 3.0}})

    def test_database_password_redacted(self):
        record = logging.LogRecord("t", logging.ERROR, "test.py", 11,
            "Cannot connect to postgresql://user:secret@localhost/db", (), None)
        text = JsonFormatter().format(record)
        self.assertNotIn("secret", text)
        self.assertIn("REDACTED", text)
        self.assertEqual(json.loads(text)["level"], "ERROR")

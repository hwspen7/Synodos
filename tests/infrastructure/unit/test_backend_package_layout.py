"""Regression for backend package consolidation."""
from __future__ import annotations

import unittest

from core.infrastructure_settings import InfrastructureSettings, load_infrastructure_settings
from core.security import origin_allowed, public_error
from core.validation import validate_application_settings
from core.logging_setup import JsonFormatter, configure_logging
from core.metrics import Metrics, metrics
from sessions.audio_buffer import AudioBuffer, AudioBackpressureError


class BackendLayoutTests(unittest.TestCase):
    def test_backend_symbols_remain_importable(self):
        self.assertTrue(callable(load_infrastructure_settings))
        self.assertTrue(callable(origin_allowed))
        self.assertTrue(callable(public_error))
        self.assertTrue(callable(validate_application_settings))
        self.assertTrue(callable(configure_logging))
        self.assertTrue(issubclass(JsonFormatter, object))
        self.assertTrue(isinstance(metrics, Metrics))
        self.assertTrue(issubclass(AudioBackpressureError, RuntimeError))
        self.assertTrue(callable(AudioBuffer))
        self.assertTrue(issubclass(InfrastructureSettings, object))

    def test_audio_capacity_contract(self):
        buffer = AudioBuffer(16000, max_seconds=2)
        self.assertEqual(buffer.capacity_samples(), 32000)
        self.assertEqual(buffer.total_samples(), 0)

    def test_metrics_contract(self):
        local = Metrics()
        local.increment("layout", 2)
        self.assertEqual(local.snapshot()["counters"]["layout"], 2)


if __name__ == "__main__":
    unittest.main()

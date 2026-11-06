"""Phase 07B bounded inference instrumentation."""
import threading
import unittest
from runtime_quality.telemetry import Phase07Telemetry, percentile


class PerformanceTelemetryTests(unittest.TestCase):
    def test_empty_percentile(self):
        self.assertIsNone(percentile([], .95))

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 100], .5), 50)

    def test_bounded_retention(self):
        telemetry = Phase07Telemetry(3)
        for i in range(50):
            telemetry.observe_inference("asr", i / 1000)
        self.assertEqual(telemetry.snapshot()["operations"]["asr"]["observations"], 3)

    def test_asr_realtime_factor(self):
        telemetry = Phase07Telemetry()
        telemetry.observe_asr_cycle(10, 2)
        self.assertEqual(telemetry.snapshot()["asr_rtf_p50"], .2)

    def test_nonfinite_and_negative_rejected(self):
        telemetry = Phase07Telemetry()
        telemetry.observe_queue_wait("asr", float('inf'))
        telemetry.observe_inference("asr", -1)
        self.assertEqual(telemetry.snapshot()["operations"], {})

    def test_queue_depth_clamped(self):
        telemetry = Phase07Telemetry()
        telemetry.observe_queue_depth(-15)
        self.assertEqual(telemetry.snapshot()["queue_depth"], 0)

    def test_multithreaded_update(self):
        telemetry = Phase07Telemetry()
        def worker():
            for _ in range(100):
                telemetry.observe_queue_wait("asr", .01)
                telemetry.observe_inference("asr", .02)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(telemetry.snapshot()["operations"]["asr"]["observations"], 256)

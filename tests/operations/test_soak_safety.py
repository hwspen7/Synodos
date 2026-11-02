"""The synthetic load harness must never target the real application port."""

import unittest
from ops.soak_test import SoakError, guard_target, pcm_frame


class SoakSafetyTests(unittest.TestCase):
    def test_permits_dedicated_local_test_service(self):
        guard_target("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/audio")

    def test_refuses_production_port(self):
        with self.assertRaises(SoakError):
            guard_target("http://127.0.0.1:8000", "ws://127.0.0.1:8000/ws/audio")

    def test_refuses_remote_host(self):
        with self.assertRaises(SoakError):
            guard_target("http://example.com:8001", "ws://example.com:8001/ws/audio")

    def test_refuses_wrong_route(self):
        with self.assertRaises(SoakError):
            guard_target("http://localhost:8001", "ws://localhost:8001/not-audio")

    def test_pcm_frame_size(self):
        self.assertEqual(len(pcm_frame(16000, 100, 0)), 3200)
        self.assertEqual(len(pcm_frame(16000, 20, 1600)), 640)


if __name__ == "__main__":
    unittest.main()

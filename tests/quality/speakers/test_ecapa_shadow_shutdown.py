"""Deterministic ECAPA shadow shutdown accounting tests.

These tests never launch SpeechBrain or a real subprocess.
"""
from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from sessions.identity.speaker_shadow import create_shadow_session


class FakeChild:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def make_shadow():
    with patch.dict(os.environ, {"MEETING_ECAPA_SHADOW": "1"}):
        shadow = create_shadow_session()
    assert shadow is not None
    shadow._start_process = lambda: FakeChild()
    return shadow


class ShadowShutdownTests(unittest.TestCase):
    def test_finished_tasks_are_counted_before_closed_message(self):
        shadow = make_shadow()
        shadow._exchange = lambda label, pcm: "seeded"
        try:
            audio = np.ones(32000, dtype=np.float32) * .2
            self.assertTrue(shadow.submit(1, audio))
            self.assertTrue(shadow.submit(2, audio))
            result = shadow.finish(wait_seconds=1)
            self.assertEqual(result["submitted"], 2)
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["discarded_on_shutdown"], 0)
            self.assertEqual(result["in_progress"], 0)
            self.assertEqual(result["remaining"], 0)
            self.assertTrue(result["closed"])
        finally:
            shadow.close()

    def test_inflight_and_queued_jobs_are_accounted_when_deadline_expires(self):
        shadow = make_shadow()
        started = threading.Event()
        release = threading.Event()
        def stall(label, pcm):
            started.set()
            release.wait(timeout=3)
            return "seeded"
        shadow._exchange = stall
        try:
            audio = np.ones(32000, dtype=np.float32) * .2
            self.assertTrue(shadow.submit(1, audio))
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(shadow.submit(1, audio))
            halfway = shadow.status()
            self.assertEqual(halfway["in_progress"], 1)
            self.assertEqual(halfway["remaining"], 1)
            result = shadow.finish(wait_seconds=0.01)
            self.assertEqual(result["submitted"], 2)
            self.assertEqual(result["completed"], 0)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["discarded_on_shutdown"], 2)
            self.assertEqual(result["in_progress"], 0)
            self.assertEqual(result["remaining"], 0)
        finally:
            release.set()
            shadow.close()

    def test_failed_task_is_distinct_from_discarded_backlog(self):
        shadow = make_shadow()
        started = threading.Event()
        release = threading.Event()
        def fail(label, pcm):
            started.set()
            release.wait(timeout=3)
            raise RuntimeError("injected model crash")
        shadow._exchange = fail
        try:
            audio = np.ones(32000, dtype=np.float32) * .2
            self.assertTrue(shadow.submit(1, audio))
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(shadow.submit(2, audio))
            release.set()
            # Wait for first failure so the second stays in the queue.
            self.assertTrue(shadow._thread is not None)
            shadow._thread.join(timeout=1)
            result = shadow.finish(wait_seconds=0)
            self.assertEqual(result["errors"], 1)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["discarded_on_shutdown"], 1)
            self.assertEqual(result["submitted"], 2)
        finally:
            release.set()
            shadow.close()


if __name__ == "__main__":
    unittest.main()

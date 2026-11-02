import threading
import unittest
from inference.scheduler import InferenceScheduler, InferenceBusyError, InferenceTimeoutError


class SchedulerTests(unittest.TestCase):
    def test_success_and_exception_propagation(self):
        q = InferenceScheduler(max_queue=2, timeout_s=3)
        try:
            self.assertEqual(q.submit("ok", lambda: 7), 7)
            with self.assertRaisesRegex(OSError, "fault"):
                q.submit("broken", lambda: (_ for _ in ()).throw(OSError("fault")))
        finally:
            q.close()

    def test_full_queue_and_running_job(self):
        q = InferenceScheduler(max_queue=1, timeout_s=5)
        started = threading.Event()
        release = threading.Event()
        results = []
        first = threading.Thread(target=lambda: results.append(q.submit("held", lambda: (started.set(), release.wait(2))[1])))
        first.start()
        self.assertTrue(started.wait(2))
        second = threading.Thread(target=lambda: results.append(q.submit("pending", lambda: 11)))
        second.start()
        try:
            # A pending task must occupy the only queue slot.
            for _ in range(10000):
                if q.pending == 1: break
                threading.Event().wait(0.0001)
            self.assertEqual(q.pending, 1)
            with self.assertRaises(InferenceBusyError):
                q.submit("overflow", lambda: 42)
        finally:
            release.set()
            first.join(3)
            second.join(3)
            q.close()
        self.assertIn(11, results)

    def test_wait_timeout_does_not_interrupt_running_work(self):
        q = InferenceScheduler(max_queue=1, timeout_s=0.01)
        started = threading.Event()
        release = threading.Event()
        try:
            with self.assertRaises(InferenceTimeoutError):
                q.submit("blocked", lambda: (started.set(), release.wait(1)), timeout_s=0.01)
            self.assertTrue(started.is_set())
        finally:
            release.set()
            q.close()

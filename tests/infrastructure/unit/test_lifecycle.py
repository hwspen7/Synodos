import threading
import unittest
from sessions.lifecycle import SessionLifecycle, SessionPhase


class SessionStateTests(unittest.TestCase):
    def test_valid_transitions_and_revisions(self):
        session = SessionLifecycle()
        self.assertEqual(session.snapshot().revision, 0)
        session.transition(SessionPhase.STARTING)
        session.transition(SessionPhase.RUNNING)
        session.stop()
        session.finish(saved=True)
        self.assertEqual(session.snapshot().phase, SessionPhase.CLOSED)
        self.assertEqual(session.snapshot().revision, 4)
        self.assertEqual(session.stop().phase, SessionPhase.CLOSED)

    def test_invalid_transition_fails(self):
        session = SessionLifecycle()
        with self.assertRaises(ValueError):
            session.transition(SessionPhase.CLOSED)

    def test_failed_save_visible(self):
        session = SessionLifecycle()
        session.transition(SessionPhase.STARTING)
        session.stop()
        session.finish(saved=False)
        self.assertEqual(session.snapshot().phase, SessionPhase.CLOSED)
        self.assertEqual(session.snapshot().error_code, "final_save_failed")

    def test_concurrent_stop_idempotent(self):
        session = SessionLifecycle()
        session.transition(SessionPhase.STARTING)
        threads = [threading.Thread(target=session.stop) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(session.snapshot().revision, 2)

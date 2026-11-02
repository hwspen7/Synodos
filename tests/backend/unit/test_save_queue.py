"""Save acknowledgements must correspond to a completed write."""

from __future__ import annotations

import threading
import unittest

from sessions.save_queue import SaveQueue


class SaveQueueTests(unittest.TestCase):
    def test_new_save_never_reuses_previous_success(self) -> None:
        queue = SaveQueue()
        first = queue.submit()
        self.assertIsNotNone(first)
        batch = queue.take()
        queue.finish(batch, True)
        self.assertTrue(first.completed.wait(0.01))
        self.assertTrue(first.succeeded)

        second = queue.submit()
        self.assertIsNotNone(second)
        self.assertFalse(second.completed.is_set())
        queue.finish(queue.take(), False)
        self.assertTrue(second.completed.wait(0.01))
        self.assertFalse(second.succeeded)

    def test_requests_arriving_during_save_wait_for_next_write(self) -> None:
        queue = SaveQueue()
        old = queue.submit()
        pending = queue.take()
        newer = queue.submit()
        queue.finish(pending, True)
        self.assertTrue(old.succeeded)
        self.assertFalse(newer.completed.is_set())
        queue.finish(queue.take(), True)
        self.assertTrue(newer.completed.wait(0.01))
        self.assertTrue(newer.succeeded)

    def test_failed_shutdown_wakes_waiters(self) -> None:
        queue = SaveQueue()
        ticket = queue.submit()
        queue.close(False)
        self.assertTrue(ticket.completed.wait(0.01))
        self.assertFalse(ticket.succeeded)
        self.assertIsNone(queue.submit())

    def test_batch_is_safe_with_concurrent_producers(self) -> None:
        queue = SaveQueue()
        tickets = []
        lock = threading.Lock()

        def produce() -> None:
            item = queue.submit()
            with lock:
                tickets.append(item)

        threads = [threading.Thread(target=produce) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        queue.finish(queue.take(), True)
        self.assertEqual(len(tickets), 16)
        self.assertTrue(all(t.completed.is_set() and t.succeeded for t in tickets))

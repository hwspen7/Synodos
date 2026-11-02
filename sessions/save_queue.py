"""A thread-safe queue of per-request durable-save acknowledgements.

Each HTTP save operation receives its own ticket. A successful write acknowledges
only tickets collected before that write began; a failure acknowledges none as
successful. This prevents an earlier save from satisfying a later request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class SaveTicket:
    completed: threading.Event = field(default_factory=threading.Event)
    succeeded: bool = False


class SaveQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[SaveTicket] = []
        self._closed = False
        self.wake_event = threading.Event()

    def submit(self) -> SaveTicket | None:
        with self._lock:
            if self._closed:
                return None
            ticket = SaveTicket()
            self._pending.append(ticket)
            self.wake_event.set()
            return ticket

    def take(self) -> list[SaveTicket]:
        with self._lock:
            if not self._pending:
                self.wake_event.clear()
                return []
            tickets = self._pending
            self._pending = []
            self.wake_event.clear()
            return tickets

    @staticmethod
    def finish(tickets: list[SaveTicket], succeeded: bool) -> None:
        for ticket in tickets:
            ticket.succeeded = succeeded
            ticket.completed.set()

    def close(self, succeeded: bool = False) -> None:
        with self._lock:
            self._closed = True
            pending = self._pending
            self._pending = []
            self.wake_event.clear()
        self.finish(pending, succeeded)

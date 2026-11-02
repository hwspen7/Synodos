"""Thread-safe, explicit session state transitions and monotonic revisions."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass


class SessionPhase(str, enum.Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    CLOSED = "closed"
    FAILED = "failed"


_ALLOWED = {
    SessionPhase.CREATED: {SessionPhase.STARTING, SessionPhase.STOPPING, SessionPhase.FAILED},
    SessionPhase.STARTING: {SessionPhase.RUNNING, SessionPhase.STOPPING, SessionPhase.FAILED},
    SessionPhase.RUNNING: {SessionPhase.STOPPING, SessionPhase.FAILED},
    SessionPhase.STOPPING: {SessionPhase.CLOSED, SessionPhase.FAILED},
    SessionPhase.FAILED: {SessionPhase.STOPPING, SessionPhase.CLOSED},
    SessionPhase.CLOSED: set(),
}


@dataclass(frozen=True)
class SessionSnapshot:
    phase: SessionPhase
    revision: int
    error_code: str | None


class SessionLifecycle:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = SessionPhase.CREATED
        self._revision = 0
        self._error: str | None = None

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(self._phase, self._revision, self._error)

    def transition(self, phase: SessionPhase, error_code: str | None = None) -> SessionSnapshot:
        with self._lock:
            if phase == self._phase:
                return self.snapshot()
            if phase not in _ALLOWED[self._phase]:
                raise ValueError(f"Invalid session transition: {self._phase.value} -> {phase.value}")
            self._phase = phase
            self._revision += 1
            if error_code:
                self._error = error_code
            return self.snapshot()

    def stop(self) -> SessionSnapshot:
        with self._lock:
            if self._phase in (SessionPhase.STOPPING, SessionPhase.CLOSED):
                return self.snapshot()
            return self.transition(SessionPhase.STOPPING)

    def finish(self, saved: bool) -> SessionSnapshot:
        with self._lock:
            if self._phase == SessionPhase.CLOSED:
                return self.snapshot()
            if not saved and self._phase != SessionPhase.FAILED:
                self.transition(SessionPhase.FAILED, "final_save_failed")
            return self.transition(SessionPhase.CLOSED)

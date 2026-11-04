"""Coordinate meeting reconnects and prevent persistent per-ID lock growth.

Locks live as long as users keep references while awaiting or holding them.
The registry itself retains only weak references to inactive locks.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any


_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()


def meeting_connection_lock(meeting_id: str) -> asyncio.Lock:
    # This function is called exclusively from the ASGI event loop.
    lock = _locks.get(meeting_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[meeting_id] = lock
    return lock


async def stop_and_join(session: Any, timeout: float = 120.0) -> bool:
    """Stop a session and join outside the event loop; report timeouts."""
    independent = getattr(session, "independent_diarization", None)
    if independent is not None:
        await asyncio.to_thread(independent.finish, min(timeout, 90.0))
    session.stop()
    worker = session.processor
    if worker.ident is None:
        return True
    await asyncio.to_thread(worker.join, timeout)
    return not worker.is_alive()

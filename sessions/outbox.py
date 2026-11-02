"""Bounded delivery to a per-session asyncio message queue.

Transcript messages are complete snapshots, so stale queued snapshots may be
replaced. This avoids accumulating unbounded blocked put coroutines when a
browser disconnects or cannot receive data fast enough.
"""

from __future__ import annotations

import asyncio


def enqueue_message(queue: asyncio.Queue[dict], message: dict) -> bool:
    """Enqueue immediately on the event loop; drop stale progress when full."""
    if queue.full():
        if message.get("type") == "state":
            return False
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(message)
        return True
    except asyncio.QueueFull:
        return False

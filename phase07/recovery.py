"""Periodic persisted transcript checkpoints for long meetings.

Only confirmed transcription state is saved through the existing MeetingStore.
This does not persist raw audio, restart a failed worker, or restore unconfirmed speech.
"""
from __future__ import annotations

import os


def autosave_seconds() -> float:
    raw = os.getenv("MEETING_PHASE07_AUTOSAVE_SECONDS", "60").strip()
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError("MEETING_PHASE07_AUTOSAVE_SECONDS must be numeric") from exc
    if seconds == 0:
        return 0.0
    if not 15 <= seconds <= 3600:
        raise ValueError("MEETING_PHASE07_AUTOSAVE_SECONDS must be 0 or 15-3600")
    return seconds

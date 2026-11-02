"""Optimistic transcript corrections, preserving provenance and revision history."""
from __future__ import annotations
from datetime import datetime, timezone


class RevisionConflict(RuntimeError):
    pass


def correct_segment(state: dict, segment_id: int, text: str, expected_revision: int) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Corrected text must contain 1 to 10000 characters")
    if expected_revision < 0:
        raise ValueError("Invalid revision")
    for segment in state.get("segments") or []:
        if int(segment.get("id", -1)) != segment_id:
            continue
        previous = int(segment.get("revision", 0))
        if previous != expected_revision:
            raise RevisionConflict(f"Stale revision: expected {expected_revision}, current {previous}")
        if segment.get("text") == text.strip():
            return segment
        segment.setdefault("corrections", []).append({
            "revision": previous,
            "text": str(segment.get("text") or ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        segment["text"] = text.strip()
        segment["revision"] = previous + 1
        segment["manually_corrected"] = True
        return segment
    raise KeyError(segment_id)

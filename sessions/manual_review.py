"""Version-safe human transcript and speaker identity corrections.

Manual evidence is applied to a SAVED meeting, never to a running model.
The original ASR text remains available, and each revision has an audit
record. This module has no dependencies on FastAPI or inference runtimes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

UNSET = object()


def original_asr_text(segment: dict[str, Any]) -> str:
    """Recover earliest raw text from historical review events when possible.

    Older saved meetings used the correction event's text field rather than
    original_asr_text. Never mistake a subsequently edited text for raw ASR.
    """
    original = segment.get("original_asr_text")
    if isinstance(original, str):
        return original
    for event in segment.get("corrections") or []:
        if not isinstance(event, dict):
            continue
        before = event.get("before")
        if isinstance(before, dict) and isinstance(before.get("text"), str):
            return before["text"]
        if isinstance(event.get("text"), str):
            return event["text"]
    return str(segment.get("text") or "")


class ReviewConflict(Exception):
    """The caller edited a stale segment revision."""


class ReviewValidationError(ValueError):
    """A review payload is invalid."""


def apply_review(
    state: dict[str, Any], segment_id: int, expected_revision: int, *,
    text: Any = UNSET,
    profile_id: Any = UNSET,
    profile_name: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Mutate a copied meeting state with an auditable human correction.

    The storage adapter must serialize read-modify-write and save the state.
    The profile ID must be checked against the enrolled speaker store by the
    caller; no raw embeddings are included in revision metadata.
    """
    if type(segment_id) is not int or segment_id < 0:
        raise ReviewValidationError("Invalid segment ID")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ReviewValidationError("Invalid expected revision")
    if text is UNSET and profile_id is UNSET:
        raise ReviewValidationError("No fields were supplied for review")
    if text is not UNSET and (
        not isinstance(text, str) or len(text) > 30000
    ):
        raise ReviewValidationError("Transcript text must be at most 30000 characters")
    if profile_id is not UNSET and (
        profile_id is not None and (
            not isinstance(profile_id, str) or len(profile_id) > 64
        )
    ):
        raise ReviewValidationError("Invalid speaker profile ID")
    if profile_id is not UNSET and profile_id is not None and (
        not isinstance(profile_name, str) or not profile_name.strip()
    ):
        raise ReviewValidationError("Speaker profile name is required")
    if not isinstance(reason, str) or len(reason) > 500:
        raise ReviewValidationError("Review reason must be at most 500 characters")

    segments = state.get("segments") or []
    segment = next(
        (s for s in segments if int(s.get("id", -1)) == segment_id), None
    )
    if segment is None:
        raise KeyError("Segment not found")
    revision = int(segment.get("revision", 0))
    if revision != expected_revision:
        raise ReviewConflict(
            f"Stale review revision {expected_revision}; current is {revision}"
        )

    before = {
        "text": str(segment.get("text") or ""),
        "speaker_id": segment.get("speaker_id"),
        "speaker_name": segment.get("speaker_name"),
        "speaker_profile_id": segment.get("speaker_profile_id"),
    }
    mapping = dict(state.get("participant_label_map") or {})
    if profile_id is UNSET:
        reviewed_local_id = before["speaker_id"]
    elif profile_id is None:
        reviewed_local_id = -1
    elif profile_id in mapping:
        reviewed_local_id = int(mapping[profile_id])
    else:
        used = {
            int(item.get("speaker_id", -1))
            for item in segments
        } | {int(value) for value in mapping.values()}
        reviewed_local_id = max({0} | {n for n in used if n > 0}) + 1

    prospective = {
        "text": before["text"] if text is UNSET else text,
        "speaker_id": reviewed_local_id,
        "speaker_name": (
            before["speaker_name"] if profile_id is UNSET
            else (profile_name.strip() if profile_id is not None
                  else "Unidentified Speaker")
        ),
        "speaker_profile_id": (
            before["speaker_profile_id"] if profile_id is UNSET
            else profile_id
        ),
    }
    if prospective == before:
        raise ReviewValidationError("Review has no observable change")

    if text is not UNSET:
        if not isinstance(segment.get("original_asr_text"), str):
            segment["original_asr_text"] = original_asr_text(segment)
        segment["text"] = text
        segment["manually_corrected"] = True

    if profile_id is not UNSET:
        segment["speaker_profile_id"] = profile_id
        segment["speaker_override"] = True
        # Speaker profile IDs are global; diarization IDs are local.
        # Assign one consistent local label per enrolled profile.
        if profile_id is not None:
            mapping[profile_id] = reviewed_local_id
            state["participant_label_map"] = mapping
        segment["speaker_override_id"] = reviewed_local_id
        segment["speaker_id"] = reviewed_local_id
        segment["speaker_override_name"] = (
            profile_name.strip() if profile_id is not None
            else "Unidentified Speaker"
        )
        segment["speaker_name"] = segment["speaker_override_name"]
        segment["speaker_status"] = "manual_override"
        segment["speaker_final_resolution_status"] = "manual_override"
        segment["speaker_uncertain"] = False
        segment["speaker_evidence_status"] = "human_review"

    after = {
        "text": str(segment.get("text") or ""),
        "speaker_id": segment.get("speaker_id"),
        "speaker_name": segment.get("speaker_name"),
        "speaker_profile_id": segment.get("speaker_profile_id"),
    }
    if after == before:
        raise ReviewValidationError("Review has no observable change")

    segment["revision"] = revision + 1
    event = {
        "revision": segment["revision"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "manual_review",
        "reason": reason.strip(),
        "before": before,
        "after": after,
    }
    segment.setdefault("corrections", []).append(event)
    return event

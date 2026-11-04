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
        old_text = str(segment.get("text") or "")
        if not isinstance(segment.get("original_asr_text"), str):
            segment["original_asr_text"] = old_text
        segment.setdefault("corrections", []).append({
            "revision": previous + 1,
            "kind": "text",
            "source": "manual_review",
            "before": {"text": old_text},
            "after": {"text": text.strip()},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        segment["text"] = text.strip()
        segment["revision"] = previous + 1
        segment["manually_corrected"] = True
        return segment
    raise KeyError(segment_id)

def assign_segment_speaker(
    state: dict, segment_id: int, participant_id: str | None,
    participant_name: str | None, expected_revision: int,
) -> dict:
    """Persist a manual speaker label across reclustering and meeting resumes.

    Assigning no participant intentionally restores the unidentified label.
    The actual speaker's identity is supplied by the operator, not inferred
    from insufficient acoustic evidence.
    """
    if expected_revision < 0:
        raise ValueError("Invalid revision")
    mapping = state.setdefault("participant_label_map", {})
    for segment in state.get("segments") or []:
        if int(segment.get("id", -1)) != segment_id:
            continue
        revision = int(segment.get("revision", 0))
        if revision != expected_revision:
            raise RevisionConflict(f"Stale revision: expected {expected_revision}, current {revision}")
        if participant_id is None:
            speaker_id, speaker_name = -1, "Unidentified Speaker"
        else:
            if not participant_name or not participant_name.strip():
                raise ValueError("A valid enrolled speaker is required")
            if participant_id not in mapping:
                used = {int(s.get("speaker_id", -1)) for s in state.get("segments") or []}
                used.update(int(v) for v in mapping.values())
                speaker_id = max({0} | {value for value in used if value > 0}) + 1
                mapping[participant_id] = speaker_id
            else:
                speaker_id = int(mapping[participant_id])
            speaker_name = participant_name.strip()
        if (segment.get("speaker_override") and
            segment.get("speaker_override_id") == speaker_id and
            segment.get("speaker_override_name") == speaker_name):
            return segment
        segment.setdefault("corrections", []).append({
            "revision": revision + 1,
            "kind": "speaker",
            "source": "manual_review",
            "before": {
                "speaker_id": segment.get("speaker_id"),
                "speaker_name": segment.get("speaker_name"),
                "speaker_profile_id": segment.get("speaker_profile_id"),
            },
            "after": {
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "speaker_profile_id": participant_id,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        segment["speaker_override"] = True
        segment["speaker_override_id"] = speaker_id
        segment["speaker_override_name"] = speaker_name
        segment["speaker_profile_id"] = participant_id
        segment["speaker_status"] = "manual_override"
        segment["speaker_evidence_status"] = "human_review"
        segment["speaker_id"] = speaker_id
        segment["speaker_name"] = speaker_name
        segment["revision"] = revision + 1
        return segment
    raise KeyError(segment_id)

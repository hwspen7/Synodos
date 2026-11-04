"""Meeting CRUD, saving, and export."""

import asyncio
import shutil
from pathlib import Path
from speech.config import load_settings
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from speech.supported_languages import spoken_language, translation_target
from fastapi import APIRouter, HTTPException, Query, Response
from core.runtime import meeting_store, meeting_sessions, sessions, speaker_store, hotword_store
from api.session_lifecycle import meeting_connection_lock, stop_and_join

from sessions.manual_review import (
    ReviewConflict, ReviewValidationError, apply_review, original_asr_text,
)
from sessions.identity_matcher import match_enrolled_voice

router = APIRouter()

class MeetingSettingsPayload(BaseModel):
    language: str = Field(default="en", min_length=2, max_length=24)
    translate_to: str = Field(default="en", max_length=24)
    record_audio: bool = True
    vad_threshold: float = Field(default=4.0, ge=0.5, le=30)
    rec_threshold: float = Field(default=1.5, ge=0.2, le=30)
    participant_ids: list[str] = Field(default_factory=list, max_length=64)
    hotword_ids: list[str] = Field(default_factory=list, max_length=64)
    spk_threshold: float = Field(default=0.3, ge=0, le=1)

    @field_validator("language")
    @classmethod
    def validate_source_language(cls, value: str) -> str:
        return spoken_language(value)

    @field_validator("translate_to")
    @classmethod
    def validate_translation_target(cls, value: str) -> str:
        return translation_target(value)


@router.patch("/api/meetings/{meeting_id}/settings")
async def update_meeting_settings(meeting_id: str, payload: MeetingSettingsPayload) -> dict:
    """Persist offline configuration without opening an idle recording session."""
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(400, "Invalid meeting ID")
    participants = speaker_store.get_for_matching(payload.participant_ids)
    if len(participants) != len(set(payload.participant_ids)):
        raise HTTPException(400, "Selected speaker is missing")
    libraries = {item["id"] for item in hotword_store.list_hotwords()}
    if any(hid not in libraries for hid in payload.hotword_ids):
        raise HTTPException(400, "Selected hotword library is missing")
    from core.runtime import settings as app_settings
    async with meeting_connection_lock(meeting_id):
        if meeting_id in meeting_sessions:
            raise HTTPException(409, "Stop the recording before updating settings")
        def save():
            meta = meeting_store.get_meta(meeting_id)
            if meta is None:
                raise HTTPException(404, "Meeting not found")
            state = meeting_store.load_state(meeting_id) or {}
            state.pop("meta", None)
            state["id"] = meeting_id
            state["title"] = state.get("title") or meta.get("title") or "Untitled Meeting"
            state.setdefault("sample_rate", app_settings.sample_rate)
            state.setdefault("merge_gap_s", app_settings.merge_gap_s)
            state["participant_ids"] = list(dict.fromkeys(payload.participant_ids))
            state["hotword_ids"] = list(dict.fromkeys(payload.hotword_ids))
            state["spk_threshold"] = payload.spk_threshold
            state["ui_settings"] = payload.model_dump()
            meeting_store.save_state(state)
            return {"ok": True, "settings": state["ui_settings"]}
        return await asyncio.to_thread(save)


@router.get("/api/meetings")
async def list_meetings() -> dict:
    meetings = await asyncio.get_running_loop().run_in_executor(None, meeting_store.list_meetings)
    return {"meetings": meetings}


@router.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> dict:
    """Return meeting metadata and participant settings to initialize the frontend."""
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    meeting = await asyncio.get_running_loop().run_in_executor(
        None, meeting_store.get_meeting, meeting_id
    )
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return {"meeting": meeting}


@router.get("/api/meetings/{meeting_id}/segments")
async def get_meeting_segments(meeting_id: str) -> dict:
    """Return segment summaries without embeddings for voiceprint enrollment."""
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    segments = await asyncio.get_running_loop().run_in_executor(
        None, meeting_store.list_segments, meeting_id
    )
    if segments is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    gaps = await asyncio.get_running_loop().run_in_executor(
        None, meeting_store.list_speech_coverage_gaps, meeting_id
    )
    return {"segments": segments, "speech_coverage_gaps": gaps or []}


@router.get("/api/meetings/{meeting_id}/segments/{segment_id}/identity-candidate")
async def get_segment_identity_candidate(
    meeting_id: str, segment_id: int,
    threshold: float = 0.3, margin: float = 0.06,
) -> dict:
    """Propose an identity, but never auto-write a tentative candidate."""
    if not meeting_store.valid_id(meeting_id) or segment_id < 0:
        raise HTTPException(400, "invalid meeting/segment id")
    state = await asyncio.to_thread(meeting_store.load_state, meeting_id)
    if state is None:
        raise HTTPException(404, "meeting not found")
    segment = next(
        (s for s in state.get("segments") or []
         if int(s.get("id", -1)) == segment_id),
        None,
    )
    if segment is None:
        raise HTTPException(404, "segment not found")
    summaries = await asyncio.to_thread(speaker_store.list_speakers)
    profiles = await asyncio.to_thread(
        speaker_store.get_for_matching, [p["id"] for p in summaries]
    )
    try:
        candidate = match_enrolled_voice(
            profiles, segment.get("emb"),
            model_id=str(segment.get("embedding_model") or "camplus"),
            threshold=threshold, margin=margin,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"segment_id": segment_id, "candidate": candidate}


@router.patch("/api/meetings/{meeting_id}/segments/{segment_id}/review")
async def review_saved_segment(
    meeting_id: str, segment_id: int, payload: dict,
) -> dict:
    """Review a stopped meeting with optimistic segment revisions.

    Saved meeting state is the source of truth for Phase 3 indexing.
    Refuse edits while a capture session could overwrite the same state.
    """
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(400, "invalid meeting id")
    if segment_id < 0 or type(payload) is not dict:
        raise HTTPException(400, "invalid review payload")
    allowed = {"expected_revision", "text", "speaker_profile_id", "reason"}
    if set(payload) - allowed:
        raise HTTPException(400, "unsupported review fields")
    expected = payload.get("expected_revision")
    if type(expected) is not int or expected < 0:
        raise HTTPException(400, "expected_revision must be a nonnegative integer")

    async with meeting_connection_lock(meeting_id):
        if meeting_id in meeting_sessions:
            raise HTTPException(409, "Stop the meeting before manual review")
        state = await asyncio.to_thread(meeting_store.load_state, meeting_id)
        if state is None:
            raise HTTPException(404, "meeting not found")
        state.pop("meta", None)
        state["id"] = meeting_id
        if not state.get("segments"):
            raise HTTPException(404, "segment not found")
        args = {
            "reason": payload.get("reason", ""),
        }
        if "text" in payload:
            args["text"] = payload["text"]
        if "speaker_profile_id" in payload:
            profile_id = payload["speaker_profile_id"]
            if profile_id is not None:
                if not isinstance(profile_id, str) or not speaker_store.valid_id(profile_id):
                    raise HTTPException(400, "invalid speaker profile id")
                profiles = await asyncio.to_thread(
                    speaker_store.get_for_matching, [profile_id]
                )
                if not profiles:
                    raise HTTPException(404, "speaker profile not found")
                args["profile_name"] = profiles[0]["name"]
            args["profile_id"] = profile_id
        try:
            event = apply_review(state, segment_id, expected, **args)
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ReviewValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "segment not found") from exc
        await asyncio.to_thread(meeting_store.save_state, state)
        segments = await asyncio.to_thread(meeting_store.list_segments, meeting_id)
        reviewed = next(
            (item for item in segments or [] if item["id"] == segment_id),
            None,
        )
        return {"segment": reviewed, "review": event}


@router.get("/api/meetings/{meeting_id}/segments/{segment_id}/revisions")
async def get_segment_revisions(meeting_id: str, segment_id: int) -> dict:
    """Read immutable review events without exposing voiceprint vectors."""
    if not meeting_store.valid_id(meeting_id) or segment_id < 0:
        raise HTTPException(400, "invalid meeting/segment id")
    state = await asyncio.to_thread(meeting_store.load_state, meeting_id)
    if state is None:
        raise HTTPException(404, "meeting not found")
    segment = next(
        (s for s in state.get("segments") or []
         if int(s.get("id", -1)) == segment_id),
        None,
    )
    if segment is None:
        raise HTTPException(404, "segment not found")
    return {
        "segment_id": segment_id,
        "revision": int(segment.get("revision", 0)),
        "original_asr_text": original_asr_text(segment),
        "corrections": list(segment.get("corrections") or []),
    }


@router.post("/api/meetings")
async def create_meeting(payload: Optional[dict] = None) -> dict:
    title = (payload or {}).get("title", "") or ""
    meta = await asyncio.get_running_loop().run_in_executor(None, meeting_store.create, None, title)
    return {"meeting": meta}


@router.post("/api/meetings/{meeting_id}/save")
async def save_meeting(meeting_id: str) -> dict:
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    session = meeting_sessions.get(meeting_id)
    if session is None:
        meta = meeting_store.get_meta(meeting_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="meeting not found")
        return {"ok": True, "meeting": meta}
    ok = await asyncio.get_running_loop().run_in_executor(
        None, session.processor.request_save, 30.0
    )
    if not ok:
        raise HTTPException(status_code=500, detail="save failed or timed out")
    return {"ok": True, "meeting": meeting_store.get_meta(meeting_id)}


@router.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> dict:
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    async with meeting_connection_lock(meeting_id):
        active = meeting_sessions.get(meeting_id)
        if active is not None:
            if not await stop_and_join(active):
                raise HTTPException(status_code=503, detail="active meeting is still finalizing")
            if meeting_sessions.get(meeting_id) is active:
                meeting_sessions.pop(meeting_id, None)
            if sessions.get(active.id) is active:
                sessions.pop(active.id, None)
            try:
                await active.ws.close(code=1000, reason="Meeting deleted")
            except Exception:
                pass
        ok = await asyncio.get_running_loop().run_in_executor(None, meeting_store.delete, meeting_id)
    if not ok:
        raise HTTPException(status_code=404, detail="meeting not found")
    # Remove the meeting's optional local audio archive after deleting the record.
    root = Path(load_settings().recording_root).resolve()
    recording_dir = (root / meeting_id).resolve()
    if recording_dir.is_relative_to(root) and recording_dir.is_dir():
        try:
            await asyncio.to_thread(shutil.rmtree, recording_dir)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Meeting deleted, but source recordings could not be removed"
            ) from exc
    return {"ok": True}


@router.get("/api/meetings/{meeting_id}/export")
async def export_meeting(
    meeting_id: str, format: str = Query(default="md")
) -> Response:
    if format not in ("md", "json"):
        raise HTTPException(status_code=400, detail="format must be md or json")
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    loop = asyncio.get_running_loop()
    session = meeting_sessions.get(meeting_id)
    if session is not None:
        # Do not export a stale snapshot after a failed save.
        saved = await loop.run_in_executor(None, session.processor.request_save, 30.0)
        if not saved:
            raise HTTPException(status_code=503, detail="latest meeting state could not be saved")
    try:
        text = await loop.run_in_executor(None, meeting_store.export, meeting_id, format)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="meeting not found")
    media_type = "application/json" if format == "json" else "text/markdown"
    filename = f"{meeting_id}.{format}"
    return Response(
        content=text,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

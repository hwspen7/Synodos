"""Capabilities, audio replay, transcription corrections, and offline translation."""
from __future__ import annotations
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import wave
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from core.runtime import meeting_store, meeting_sessions, speaker_store
from speech.config import load_settings
from speech.captions.corrections import correct_segment, assign_segment_speaker, RevisionConflict
from speech.captions.translation import translate, TranslationUnavailable
from speech.supported_languages import translation_target
from sessions.translation_tasks import CaptionTranslation, caption_batch, submit_translation

router = APIRouter(prefix="/api/speech", tags=["speech"])


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source_language: str = Field(min_length=2, max_length=24)
    target_language: str = Field(min_length=2, max_length=24)


class SavedTranslationRequest(BaseModel):
    target_language: str = Field(min_length=2, max_length=24)
    after_id: int = Field(default=-1, ge=-1)


class CorrectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    revision: int = Field(default=0, ge=0)


class SpeakerCorrectionRequest(BaseModel):
    participant_id: str | None = None
    revision: int = Field(default=0, ge=0)


@router.get("/capabilities")
async def capabilities() -> dict:
    cfg = load_settings()
    return {"engine": cfg.engine, "model": cfg.model,
            "language": cfg.language, "translation_target": cfg.translate_to,
            "audio_recording": cfg.record_audio,
            "native_streaming_configured": cfg.native_streaming,
            "timestamps": cfg.engine in ("whisper", "qwen3"),
            "overlap_detection_configured": cfg.enable_overlap,
            "denoising_configured": cfg.enable_denoise,
            "adaptive_asr_audio_enabled": cfg.adaptive_audio_enabled,
            "event_detection_configured": cfg.enable_events}


@router.post("/translate")
async def translate_text(payload: TranslationRequest) -> dict:
    try:
        result = await asyncio.to_thread(translate, payload.text, payload.source_language, payload.target_language)
    except TranslationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"text": result, "source_language": payload.source_language,
            "target_language": payload.target_language}


@router.post("/meetings/{meeting_id}/translations")
async def translate_saved_meeting(meeting_id: str, payload: SavedTranslationRequest) -> dict:
    """Translate one bounded page, then persist only still-current captions."""
    from api.session_lifecycle import meeting_connection_lock
    from sessions.manual_review import ReviewConflict

    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(400, "Invalid meeting ID")
    try:
        target = translation_target(payload.target_language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    async def load_saved() -> dict:
        session = meeting_sessions.get(meeting_id)
        if session is not None and not (session.processor.stopped and session.processor._final_save_ok):
            raise HTTPException(409, "Stop recording before translating saved captions")
        state = await asyncio.to_thread(meeting_store.load_state, meeting_id)
        if state is None:
            raise HTTPException(404, "Meeting not found")
        if (state.get("ui_settings") or {}).get("translate_to") != target:
            raise HTTPException(409, "Save the selected translation language first")
        return state

    async with meeting_connection_lock(meeting_id):
        state = await load_saved()
        pending = sorted(
            (CaptionTranslation.from_segment(segment) for segment in state.get("segments") or []
             if int(segment["id"]) > payload.after_id
             and segment.get("language") != target
             and not (segment.get("translation_target") == target and segment.get("translation"))),
            key=lambda caption: caption.id,
        )
    batch = caption_batch(pending)
    if not batch:
        return {"attempted": 0, "translated": 0, "next_after": payload.after_id, "has_more": False}
    future = submit_translation(batch, target)
    if future is None:
        raise HTTPException(503, "Translation queue is busy. Try again shortly.")
    try:
        translations = await asyncio.wrap_future(future)
    except TranslationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    async with meeting_connection_lock(meeting_id):
        state = await load_saved()
        by_id = {int(segment["id"]): segment for segment in state.get("segments") or []}
        count = sum(
            caption.apply(by_id[caption.id], target, translations.get(caption.id, ""))
            for caption in batch if caption.id in by_id
        )
        if count:
            revision = (state.pop("meta", {}) or {}).get("_store_version")
            try:
                await asyncio.to_thread(meeting_store.save_state, state,
                                        expected_store_revision=revision)
            except ReviewConflict as exc:
                raise HTTPException(409, str(exc)) from exc
    return {"attempted": len(batch), "translated": count, "next_after": batch[-1].id,
            "has_more": len(pending) > len(batch)}


@router.patch("/meetings/{meeting_id}/segments/{segment_id}")
async def edit_segment(meeting_id: str, segment_id: int, payload: CorrectionRequest) -> dict:
    if not meeting_store.valid_id(meeting_id) or segment_id < 0:
        raise HTTPException(400, "Invalid meeting or segment ID")
    if meeting_id in meeting_sessions:
        raise HTTPException(409, "Stop the meeting before editing confirmed text")
    from api.session_lifecycle import meeting_connection_lock
    async with meeting_connection_lock(meeting_id):
        if meeting_id in meeting_sessions:
            raise HTTPException(409, "Meeting is active")
        def edit():
            state = meeting_store.load_state(meeting_id)
            if state is None:
                raise HTTPException(404, "Meeting not found")
            try:
                result = correct_segment(state, segment_id, payload.text, payload.revision)
            except RevisionConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(404, "Segment not found") from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            meeting_store.save_state(state)
            return {"ok": True, "segment_id": segment_id, "revision": result["revision"], "text": result["text"]}
        return await asyncio.to_thread(edit)


@router.patch("/meetings/{meeting_id}/segments/{segment_id}/speaker")
async def edit_segment_speaker(
    meeting_id: str, segment_id: int, payload: SpeakerCorrectionRequest,
) -> dict:
    if not meeting_store.valid_id(meeting_id) or segment_id < 0:
        raise HTTPException(400, "Invalid meeting or segment ID")
    from api.session_lifecycle import meeting_connection_lock
    async with meeting_connection_lock(meeting_id):
        if meeting_id in meeting_sessions:
            raise HTTPException(409, "Stop the recording before correcting the speaker")
        speaker = None
        if payload.participant_id is not None:
            if not speaker_store.valid_id(payload.participant_id):
                raise HTTPException(400, "Invalid participant ID")
            found = speaker_store.get_for_matching([payload.participant_id])
            if not found:
                raise HTTPException(404, "Selected speaker does not exist")
            speaker = found[0]

        def edit():
            state = meeting_store.load_state(meeting_id)
            if state is None:
                raise HTTPException(404, "Meeting not found")
            try:
                result = assign_segment_speaker(
                    state, segment_id, payload.participant_id,
                    speaker["name"] if speaker else None, payload.revision,
                )
            except RevisionConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(404, "Segment not found") from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            meeting_store.save_state(state)
            projected = next(
                row for row in meeting_store.list_segments(meeting_id) or ()
                if row["id"] == segment_id
            )
            return {
                "speaker_id": projected["speaker_id"],
                "speaker_name": projected["speaker_name"],
                "speaker_final_resolution_status": projected["speaker_final_resolution_status"],
                "revision": result["revision"],
            }

        return await asyncio.to_thread(edit)


@router.get("/meetings/{meeting_id}/recordings")
async def list_recordings(meeting_id: str) -> dict:
    cfg = load_settings()
    if not meeting_store.valid_id(meeting_id) or meeting_store.get_meta(meeting_id) is None:
        raise HTTPException(404, "Meeting not found")
    folder = Path(cfg.recording_root).resolve() / meeting_id
    def describe(path: Path) -> dict:
        stat = path.stat()
        duration = None
        try:
            with wave.open(str(path), "rb") as source:
                duration = round(source.getnframes() / source.getframerate(), 2)
        except (OSError, EOFError, ValueError, wave.Error, ZeroDivisionError):
            pass
        return {
            "name": path.name,
            "bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "duration_s": duration,
        }
    return {"recordings": [describe(p) for p in sorted(
        folder.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True,
    )]} if folder.is_dir() else {"recordings": []}


@router.get("/meetings/{meeting_id}/recordings/{filename}")
async def play_recording(meeting_id: str, filename: str):
    cfg = load_settings()
    if not meeting_store.valid_id(meeting_id) or meeting_store.get_meta(meeting_id) is None:
        raise HTTPException(404, "Meeting not found")
    import re
    if not re.fullmatch(r"[a-f0-9]{32}\.wav", filename):
        raise HTTPException(400, "Invalid recording name")
    path = Path(cfg.recording_root).resolve() / meeting_id / filename
    if not path.is_file():
        raise HTTPException(404, "Recording not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)

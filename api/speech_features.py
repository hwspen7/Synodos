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
from speech.corrections import correct_segment, assign_segment_speaker, RevisionConflict
from speech.translation import translate, TranslationUnavailable

router = APIRouter(prefix="/api/speech", tags=["speech"])


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source_language: str = Field(min_length=2, max_length=24)
    target_language: str = Field(min_length=2, max_length=24)


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
            "event_detection_configured": cfg.enable_events}


@router.post("/translate")
async def translate_text(payload: TranslationRequest) -> dict:
    try:
        result = await asyncio.to_thread(translate, payload.text, payload.source_language, payload.target_language)
    except TranslationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"text": result, "source_language": payload.source_language,
            "target_language": payload.target_language}


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
            return {
                "speaker_id": result["speaker_id"],
                "speaker_name": result["speaker_name"],
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

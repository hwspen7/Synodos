"""Meeting CRUD, saving, and export."""

import asyncio
import shutil
from pathlib import Path
from speech.config import load_settings
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from core.runtime import meeting_store, meeting_sessions, sessions
from api.session_lifecycle import meeting_connection_lock, stop_and_join

router = APIRouter()

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
    return {"segments": segments}


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

"""Meeting CRUD, saving, and export."""

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from core.runtime import meeting_store, meeting_sessions

router = APIRouter()

@router.get("/api/meetings")
async def list_meetings() -> dict:
    meetings = await asyncio.get_running_loop().run_in_executor(None, meeting_store.list_meetings)
    return {"meetings": meetings}


@router.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> dict:
    """会议摘要 + 参与者设置，供前端初始化“会议参与者”选择。"""
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
    """返回会议片段摘要（不含声纹向量），供导入说话人声纹时选择。"""
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
    ok = await asyncio.get_running_loop().run_in_executor(None, meeting_store.delete, meeting_id)
    if not ok:
        raise HTTPException(status_code=404, detail="meeting not found")
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
        # 导出前先落盘最新状态
        await loop.run_in_executor(None, session.processor.request_save, 30.0)
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

"""Hotword libraries."""

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from core.runtime import hotword_store

router = APIRouter()

@router.get("/api/hotwords")
async def list_hotwords() -> dict:
    libraries = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.list_hotwords
    )
    return {"hotwords": libraries}


@router.post("/api/hotwords")
async def create_hotword_library(payload: Optional[dict] = None) -> dict:
    name = (payload or {}).get("name", "") or ""
    words = (payload or {}).get("words") or []
    library = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.create, name, words
    )
    return {"hotword": library}


@router.patch("/api/hotwords/{hotword_id}")
async def update_hotword_library(hotword_id: str, payload: Optional[dict] = None) -> dict:
    if not hotword_store.valid_id(hotword_id):
        raise HTTPException(status_code=400, detail="invalid hotword id")
    name = (payload or {}).get("name")
    words = (payload or {}).get("words")
    library = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.update, hotword_id, name, words
    )
    if library is None:
        raise HTTPException(status_code=404, detail="hotword library not found")
    return {"hotword": library}


@router.delete("/api/hotwords/{hotword_id}")
async def delete_hotword_library(hotword_id: str) -> dict:
    if not hotword_store.valid_id(hotword_id):
        raise HTTPException(status_code=400, detail="invalid hotword id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.delete, hotword_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="hotword library not found")
    return {"ok": True}

"""Speaker enrollment and voiceprint operations."""

import asyncio
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from core.runtime import speaker_store, meeting_store, settings, models

router = APIRouter()

@router.get("/api/speakers")
async def list_speakers() -> dict:
    speakers = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.list_speakers
    )
    return {"speakers": speakers}


@router.post("/api/speakers")
async def create_speaker(payload: Optional[dict] = None) -> dict:
    name = (payload or {}).get("name", "") or ""
    speaker = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.create, name
    )
    return {"speaker": speaker}


@router.patch("/api/speakers/{speaker_id}")
async def rename_speaker(speaker_id: str, payload: Optional[dict] = None) -> dict:
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    name = (payload or {}).get("name", "") or ""
    speaker = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.rename, speaker_id, name
    )
    if speaker is None:
        raise HTTPException(status_code=404, detail="speaker not found")
    return {"speaker": speaker}


@router.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str) -> dict:
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.delete_speaker, speaker_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="speaker not found")
    return {"ok": True}


@router.post("/api/speakers/{speaker_id}/vectors/record")
async def add_speaker_record(speaker_id: str, request: Request) -> dict:
    """Extract a voiceprint from 16 kHz mono signed 16-bit little-endian PCM."""
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    body = await request.body()
    if len(body) < 2:
        raise HTTPException(status_code=400, detail="No audio data received")
    pcm = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    duration_s = len(pcm) / settings.sample_rate
    if duration_s < 0.2:
        raise HTTPException(status_code=400, detail="Recording is too short to extract a voiceprint")
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, models.embed, pcm)
    if emb is None:
        raise HTTPException(status_code=400, detail="Unable to extract a voiceprint from this recording")
    vector = {
        "emb": emb,
        "duration_s": duration_s,
        "source": "recording",
        "short": duration_s < settings.min_speaker_audio_s,
    }
    added = await loop.run_in_executor(
        None, speaker_store.add_vectors, speaker_id, [vector]
    )
    if added is None:
        raise HTTPException(status_code=404, detail="speaker not found")
    return {
        "vectors": added,
        "min_audio_s": settings.min_speaker_audio_s,
    }


@router.post("/api/speakers/{speaker_id}/vectors/meeting")
async def add_speaker_from_meeting(speaker_id: str, payload: Optional[dict] = None) -> dict:
    """Enroll one or more voiceprints from segments of a saved meeting."""
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    meeting_id = (payload or {}).get("meeting_id") or ""
    segment_ids = [int(s) for s in ((payload or {}).get("segment_ids") or [])]
    if not meeting_id or not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    if not segment_ids:
        raise HTTPException(status_code=400, detail="Select at least one segment")

    loop = asyncio.get_running_loop()
    state = await loop.run_in_executor(None, meeting_store.load_state, meeting_id)
    if state is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    segs = state.get("segments") or []
    by_id = {int(s["id"]): s for s in segs}
    sr = int(state.get("sample_rate", 16000))

    vectors: list[dict] = []
    missing: list[int] = []
    for sid in segment_ids:
        seg = by_id.get(sid)
        if seg is None:
            missing.append(sid)
            continue
        emb = np.asarray(seg.get("emb"))
        if emb is None or emb.size == 0:
            missing.append(sid)
            continue
        duration_s = (int(seg["end_sample"]) - int(seg["start_sample"])) / sr
        vectors.append(
            {
                "emb": emb,
                "duration_s": duration_s,
                "source": "meeting",
                "meeting_id": meeting_id,
                "segment_ids": [sid],
                "text": seg.get("text", ""),
                "short": duration_s < settings.min_speaker_audio_s,
            }
        )
    if not vectors:
        raise HTTPException(status_code=400, detail="Selected segments have no usable voiceprint embeddings")

    added = await loop.run_in_executor(
        None, speaker_store.add_vectors, speaker_id, vectors
    )
    if added is None:
        raise HTTPException(status_code=404, detail="speaker not found")
    return {
        "vectors": added,
        "missing": missing,
        "min_audio_s": settings.min_speaker_audio_s,
    }


@router.delete("/api/speakers/{speaker_id}/vectors/{vector_id}")
async def delete_speaker_vector(speaker_id: str, vector_id: str) -> dict:
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.delete_vector, speaker_id, vector_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="vector not found")
    return {"ok": True}

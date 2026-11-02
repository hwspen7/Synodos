"""Realtime WebSocket streaming and connection lifecycle."""

import asyncio
import contextlib
import json
import time
import uuid
from typing import Optional
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from core.runtime import settings, sessions, meeting_sessions, meeting_store, speaker_store, hotword_store
from sessions.session import Session
from api.session_lifecycle import meeting_connection_lock, stop_and_join
from api.socket_transport import decode_pcm_frame
from audio.buffer import AudioBackpressureError
from core.runtime import infra
from foundation.security import origin_allowed
from observability.metrics import metrics
from sessions.lifecycle import SessionPhase

router = APIRouter()

async def _outbox_forwarder(session: Session) -> None:
    """Forward messages from the processor thread to the client in order."""
    while True:
        payload = await session.outbox.get()
        try:
            await session.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            # Stop forwarding after client disconnection; processor messages may be dropped.
            return

@router.websocket("/ws/audio")
async def ws_audio(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default=None),
    vad_threshold: Optional[float] = Query(default=None),
    rec_threshold: Optional[float] = Query(default=None),
    meeting_id: Optional[str] = Query(default=None),
    title: Optional[str] = Query(default=None),
    participants: Optional[str] = Query(default=None),
    hotwords: Optional[str] = Query(default=None),
    spk_threshold: Optional[float] = Query(default=None),
):
    if not origin_allowed(websocket.headers.get("origin"), infra):
        metrics.increment("websocket_origin_rejected")
        await websocket.close(code=1008, reason="Origin not permitted")
        return
    if len(sessions) >= infra.max_websocket_connections:
        metrics.increment("websocket_capacity_rejected")
        await websocket.close(code=1013, reason="Session capacity reached")
        return
    await websocket.accept()
    metrics.increment("websocket_accepted")

    vad_s = float(vad_threshold if vad_threshold and vad_threshold > 0 else settings.vad_threshold_s)
    rec_s = float(rec_threshold if rec_threshold and rec_threshold > 0 else settings.rec_threshold_s)
    vad_s = max(0.5, min(vad_s, 30.0))
    rec_s = max(0.2, min(rec_s, 30.0))

    loop = asyncio.get_running_loop()
    # Serialize meeting handover, snapshot reload and new registration.
    connection_lock = meeting_connection_lock(meeting_id) if meeting_id else None
    if connection_lock is not None:
        await connection_lock.acquire()
    try:
        loaded_state: Optional[dict] = None
        meeting_title = title or ""
        if meeting_id:
            if not meeting_store.valid_id(meeting_id):
                await websocket.close(code=1008)
                return
            # Finalize the previous recording before reading persisted state.
            old = meeting_sessions.get(meeting_id)
            if old is not None:
                if not await stop_and_join(old):
                    await websocket.close(code=1013, reason="Previous session is still finalizing")
                    return
                if meeting_sessions.get(meeting_id) is old:
                    meeting_sessions.pop(meeting_id, None)
                if sessions.get(old.id) is old:
                    sessions.pop(old.id, None)
                with contextlib.suppress(Exception):
                    await old.ws.close(code=1012, reason="Meeting session replaced")
            if meeting_store.exists(meeting_id):
                loaded_state = await loop.run_in_executor(None, meeting_store.load_state, meeting_id)
                meta = (loaded_state or {}).get("meta") or {}
                meeting_title = meta.get("title") or meeting_title
            else:
                # Create the meeting record now; persist transcription state on save or stop.
                await loop.run_in_executor(None, meeting_store.create, meeting_id, title or "")

        # Explicitly selected participants take precedence over saved settings.
        if participants is not None:
            participant_ids = [p.strip() for p in participants.split(",") if p.strip()]
        else:
            participant_ids = list((loaded_state or {}).get("participant_ids") or [])
        participants_explicit = participants is not None
        # Explicitly selected hotword libraries take precedence over saved settings.
        if hotwords is not None:
            hotword_ids = [h.strip() for h in hotwords.split(",") if h.strip()]
        else:
            hotword_ids = list((loaded_state or {}).get("hotword_ids") or [])
        hotwords_explicit = hotwords is not None
        if spk_threshold is not None:
            match_thr = float(spk_threshold)
        else:
            match_thr = float((loaded_state or {}).get("spk_threshold", settings.spk_match_threshold))
        match_thr = max(0.0, min(match_thr, 1.0))
        spk_threshold_explicit = spk_threshold is not None

        # Never overwrite another active connection sharing a client-provided ID.
        sid = session_id if session_id and session_id not in sessions else uuid.uuid4().hex

        session = Session(
            sid,
            websocket,
            vad_s,
            rec_s,
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            meeting_store=meeting_store,
            speaker_store=speaker_store,
            hotword_store=hotword_store,
            participant_ids=participant_ids,
            hotword_ids=hotword_ids,
            spk_threshold=match_thr,
            participants_explicit=participants_explicit,
            hotwords_explicit=hotwords_explicit,
            spk_threshold_explicit=spk_threshold_explicit,
        )
        sessions[sid] = session
        if meeting_id:
            meeting_sessions[meeting_id] = session
        try:
            session.start(loop, state=loaded_state)
        except Exception:
            if sessions.get(sid) is session:
                sessions.pop(sid, None)
            if meeting_id and meeting_sessions.get(meeting_id) is session:
                meeting_sessions.pop(meeting_id, None)
            raise

    finally:
        if connection_lock is not None:
            connection_lock.release()

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": sid,
            "sample_rate": settings.sample_rate,
            "vad_threshold_s": vad_s,
            "rec_threshold_s": rec_s,
            "merge_gap_s": settings.merge_gap_s,
            "meeting_id": meeting_id,
            "resumed": bool(loaded_state and loaded_state.get("segments")),
            "title": session.processor.meeting_title,
            "participant_ids": session.processor.participant_ids,
            "participants": [
                {"id": p["id"], "name": p["name"]} for p in session.processor._participants
            ],
            "hotword_ids": session.processor.hotword_ids,
            "hotwords": hotword_store.get_summaries(session.processor.hotword_ids),
            "spk_threshold": session.processor.spk_threshold,
            "min_speaker_audio_s": settings.min_speaker_audio_s,
        }
    )

    forwarder = asyncio.create_task(_outbox_forwarder(session))
    stop_requested = False
    try:
        while True:
            message = await websocket.receive()
            kind = message.get("type")
            if kind == "websocket.receive":
                data = message.get("bytes")
                if data:
                    try:
                        session.buffer.append(decode_pcm_frame(data))
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                    except AudioBackpressureError:
                        metrics.increment("websocket_backpressure_stops")
                        await websocket.send_json({"type": "error", "message": "Audio buffer limit reached; recording is stopping"})
                        break
                text = message.get("text")
                if text:
                    try:
                        ctrl = json.loads(text)
                    except json.JSONDecodeError:
                        ctrl = {}
                    if ctrl.get("type") == "stop":
                        # Finish processing the remaining buffered audio before closing.
                        stop_requested = True
                        break
            elif kind == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Trigger final transcription and saving, then wait for outgoing messages.
        session.stop()
        # Join in an executor to avoid blocking WebSocket heartbeats.
        await asyncio.get_running_loop().run_in_executor(None, session.processor.join, 120)
        # Give the forwarder time to deliver the final transcript and closed message.
        drain_deadline = time.monotonic() + 5.0
        while not session.outbox.empty() and time.monotonic() < drain_deadline:
            await asyncio.sleep(0.05)
        forwarder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await forwarder
        if sessions.get(sid) is session:
            sessions.pop(sid, None)
        if meeting_id and meeting_sessions.get(meeting_id) is session:
            meeting_sessions.pop(meeting_id, None)
        if not session.processor.is_alive():
            session.lifecycle.finish(
                saved=not bool(meeting_id) or bool(session.processor._final_save_ok)
            )
        else:
            metrics.increment("websocket_finalize_timeouts")
            session.lifecycle.transition(SessionPhase.FAILED, "join_timeout")
        if stop_requested:
            try:
                await websocket.close()
            except Exception:
                pass

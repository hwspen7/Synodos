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
from speech.config import load_settings
from speech.diarization import resolve_diarization_mode
from speech.recording import RecordingWriter, RecordingQuotaExceeded
from speech.streaming import NativeOnlineRecognizer, PreviewTracker
import os
from api.session_lifecycle import meeting_connection_lock, stop_and_join
from api.socket_transport import decode_pcm_frame, AudioFrameSequence, AudioSequenceError
from audio.buffer import AudioBackpressureError
from core.runtime import infra
from foundation.security import origin_allowed
from observability.metrics import metrics
from sessions.lifecycle import SessionPhase

router = APIRouter()


def resolve_record_audio(
    requested: bool | None, saved_state: dict | None,
    server_default: bool,
) -> bool:
    """Respect explicit consent first, then the saved meeting preference."""
    if requested is not None:
        return bool(requested)
    saved = (saved_state or {}).get("ui_settings")
    value = saved.get("record_audio") if isinstance(saved, dict) else None
    return value if type(value) is bool else bool(server_default)


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
    language: str = Query(default="en"),
    translate_to: Optional[str] = Query(default=None),
    record_audio: Optional[bool] = Query(default=None),
):
    if not origin_allowed(websocket.headers.get("origin"), infra):
        metrics.increment("websocket_origin_rejected")
        await websocket.close(code=1008, reason="Origin not permitted")
        return
    if len(sessions) >= infra.max_websocket_connections:
        metrics.increment("websocket_capacity_rejected")
        await websocket.close(code=1013, reason="Session capacity reached")
        return
    speech_config = load_settings()
    if len(language) > 24 or len(translate_to or "") > 24:
        await websocket.close(code=1008, reason="Invalid language")
        return
    await websocket.accept()
    metrics.increment("websocket_accepted")
    audio_archive = None
    online = None
    online_preview = PreviewTracker()
    if speech_config.native_streaming:
        try:
            online = NativeOnlineRecognizer(
                os.environ.get("MEETING_ONLINE_ENCODER", ""),
                os.environ.get("MEETING_ONLINE_DECODER", ""),
                os.environ.get("MEETING_ONLINE_JOINER", ""),
                os.environ.get("MEETING_ONLINE_TOKENS", ""),
            )
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Native streaming initialization failed: {exc}"})
            await websocket.close(code=1011)
            return

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
        # A saved meeting's recording preference must not be silently
        # overridden by the server-wide default on reconnect. Explicit
        # connection-level consent always takes precedence.
        record_audio_selected = resolve_record_audio(
            record_audio, loaded_state, speech_config.record_audio
        )
        try:
            session.processor.speech_language = language
            session.processor.translate_target = speech_config.translate_to if translate_to is None else translate_to
            session.start(loop, state=loaded_state)
            session.processor.ui_settings = {
                "language": language,
                "translate_to": session.processor.translate_target,
                "record_audio": record_audio_selected,
                "vad_threshold": vad_s,
                "rec_threshold": rec_s,
                "participant_ids": list(session.processor.participant_ids),
                "hotword_ids": list(session.processor.hotword_ids),
                "spk_threshold": session.processor.spk_threshold,
            }
        except Exception:
            if sessions.get(sid) is session:
                sessions.pop(sid, None)
            if meeting_id and meeting_sessions.get(meeting_id) is session:
                meeting_sessions.pop(meeting_id, None)
            raise

    finally:
        if connection_lock is not None:
            connection_lock.release()

    # Initialize the WAV writer before announcing whether recording is active.
    recording_requested = bool(meeting_id) and record_audio_selected
    archive_warning = None
    if recording_requested:
        try:
            audio_archive = RecordingWriter(
                speech_config.recording_root, meeting_id,
                speech_config.max_recording_bytes,
            )
        except Exception as exc:
            archive_warning = f"WAV recording could not start: {exc}"

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
            "participant_enrollments": [
                {"id": item["id"], "name": item["name"], "voiceprint_count": len(item["vectors"])}
                for item in session.processor._participants
            ],
            "participants": [
                {"id": p["id"], "name": p["name"]} for p in session.processor._participants
            ],
            "hotword_ids": session.processor.hotword_ids,
            "active_hotword_count": session.processor.active_hotword_count,
            "processing_window_s": session.processor._process_threshold_s,
            "confirmation_delay_s": session.processor._trusted_threshold_s,
            "min_commit_context_s": getattr(session.processor, "_min_commit_context_s", 0),
            "hotwords": hotword_store.get_summaries(session.processor.hotword_ids),
            "spk_threshold": session.processor.spk_threshold,
            "min_speaker_audio_s": settings.min_speaker_audio_s,
            "asr_engine": speech_config.engine,
            "asr_model": speech_config.model if speech_config.engine != "legacy" else settings.asr_model,
            "diarization_mode": resolve_diarization_mode(
                os.getenv("MEETING_DIARIZATION_MODE")
            ),
            "asr_language": session.processor.speech_language,
            "translation_target": session.processor.translate_target,
            "wav_recording_enabled": audio_archive is not None,
        }
    )
    if archive_warning:
        await websocket.send_json({
            "type": "feature_warning", "feature": "recording",
            "message": archive_warning,
        })
    forwarder = asyncio.create_task(_outbox_forwarder(session))
    stop_requested = False
    received_samples = 0
    frame_sequence = AudioFrameSequence()
    last_progress_samples = 0
    try:
        while True:
            message = await websocket.receive()
            kind = message.get("type")
            if kind == "websocket.receive":
                data = message.get("bytes")
                if data:
                    try:
                        frame, duplicate = frame_sequence.accept(data)
                        if duplicate:
                            await session.outbox.put({
                                "type": "audio_progress",
                                "samples": received_samples,
                                "last_sequence": frame_sequence.next_sequence - 1,
                            })
                            continue
                        assert frame is not None
                        session.buffer.append(frame)
                        received_samples += len(frame)
                        if audio_archive is not None:
                            # Keep the WAV archive in raw PCM even when the
                            # transport uses the versioned frame header.
                            audio_archive.append(
                                data[16:] if frame_sequence.versioned else data
                            )
                        # A running microphone UI must be backed by confirmed
                        # server-side PCM, not just by browser permissions.
                        if received_samples - last_progress_samples >= settings.sample_rate // 2:
                            last_progress_samples = received_samples
                            await session.outbox.put({
                                "type": "audio_progress",
                                "samples": received_samples,
                                "durable_samples": audio_archive.durable_bytes // 2 if audio_archive is not None else None,
                                "last_sequence": frame_sequence.next_sequence - 1 if frame_sequence.versioned else None,
                            })
                        if online is not None:
                            live_text = await asyncio.to_thread(online.accept, frame)
                            preview = online_preview.update(live_text, language, 0,
                                                            session.buffer.absolute_end_samples() * 1000 // settings.sample_rate)
                            if preview is not None:
                                await session.outbox.put({"type": "partial", **vars(preview), "native": True})
                    except AudioSequenceError as exc:
                        metrics.increment("audio_sequence_errors")
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        break
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        break
                    except RecordingQuotaExceeded as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        break
                    except OSError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "WAV recording failed while writing audio. Check disk space and permissions.",
                        })
                        break
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
                        expected = ctrl.get("expected_samples")
                        if expected is not None and (
                            type(expected) is not int or expected < 0 or expected != received_samples
                        ):
                            metrics.increment("audio_sample_count_mismatches")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Audio delivery mismatch. Some microphone samples may be missing.",
                            })
                        durable_samples = None
                        if audio_archive is not None:
                            try:
                                durable_samples = audio_archive.checkpoint()
                            except OSError as exc:
                                metrics.increment("wav_checkpoint_failure")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"WAV audio checkpoint failed: {type(exc).__name__}",
                                })
                                break
                        await websocket.send_json({
                            "type": "audio_received",
                            "samples": received_samples,
                            "durable_samples": durable_samples,
                            "matched": expected is None or (
                                type(expected) is int and expected == received_samples
                            ),
                            "last_sequence": frame_sequence.next_sequence - 1 if frame_sequence.versioned else None,
                        })
                        stop_requested = True
                        break
            elif kind == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if audio_archive is not None:
            try:
                audio_archive.close()
            except OSError as exc:
                metrics.increment("wav_close_failure")
                await session.outbox.put({
                    "type": "error",
                    "message": f"WAV finalization failed: {type(exc).__name__}",
                })
        if online is not None:
            try:
                await asyncio.to_thread(online.finish)
            except Exception:
                pass
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

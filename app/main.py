"""FastAPI 入口：WebSocket 实时音频流 + 会议记录管理（保存/续记/导出）。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .buffer import AudioBuffer
from .config import Settings, make_settings
from .hotword_store import HotwordStore
from .models import ModelManager
from .processor import SessionProcessor
from .speaker_store import SpeakerStore
from .store import MeetingStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("meeting")

app = FastAPI(title="Meeting of Pioneers", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = make_settings()
models = ModelManager(settings)
meeting_store = MeetingStore(settings.meetings_dir)
speaker_store = SpeakerStore(settings.speakers_dir)
hotword_store = HotwordStore(settings.hotwords_dir)
sessions: dict[str, "Session"] = {}
# 正在直播的会议（meeting_id -> Session），用于保存/导出时拿最新状态
meeting_sessions: dict[str, "Session"] = {}


class Session:
    """一个 WebSocket 连接对应的会话：音频缓冲 + 全量音频 + 处理线程 + 出站队列。"""

    def __init__(
        self,
        session_id: str,
        ws: WebSocket,
        vad_threshold_s: float,
        rec_threshold_s: float,
        meeting_id: Optional[str] = None,
        meeting_title: str = "",
        meeting_store: Optional[MeetingStore] = None,
        speaker_store: Optional[SpeakerStore] = None,
        hotword_store: Optional[HotwordStore] = None,
        participant_ids: Optional[list[str]] = None,
        hotword_ids: Optional[list[str]] = None,
        spk_threshold: Optional[float] = None,
        participants_explicit: bool = False,
        hotwords_explicit: bool = False,
        spk_threshold_explicit: bool = False,
    ):
        self.id = session_id
        self.ws = ws
        self.meeting_id = meeting_id
        self.settings = Settings(
            sample_rate=settings.sample_rate,
            vad_threshold_s=vad_threshold_s,
            rec_threshold_s=rec_threshold_s,
            merge_gap_s=settings.merge_gap_s,
            final_discard_s=settings.final_discard_s,
            cluster_merge_thr=settings.cluster_merge_thr,
            asr_model=settings.asr_model,
            fsmn_vad_model=settings.fsmn_vad_model,
            punc_model=settings.punc_model,
            spk_model=settings.spk_model,
        )
        self.buffer = AudioBuffer(settings.sample_rate)
        self.outbox: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)
        self.processor = SessionProcessor(
            session_id=session_id,
            buffer=self.buffer,
            settings=self.settings,
            models=models,
            outbox=self.outbox,
            meeting_store=meeting_store,
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            speaker_store=speaker_store,
            participant_ids=participant_ids,
            hotword_store=hotword_store,
            hotword_ids=hotword_ids,
            spk_threshold=spk_threshold,
            participants_explicit=participants_explicit,
            hotwords_explicit=hotwords_explicit,
            spk_threshold_explicit=spk_threshold_explicit,
        )

    def start(self, loop: asyncio.AbstractEventLoop, state: Optional[dict] = None) -> None:
        self.processor.set_loop(loop)
        if state is not None:
            self.processor.load_state(state)
        self.processor.load_participants()
        self.processor.load_hotwords()
        # 恢复历史段后按当前参与者/阈值重算标签，保证续记与会话内逻辑一致
        self.processor.rebuild_labels()
        self.processor.start()

    def stop(self) -> None:
        self.processor.stop()


async def _outbox_forwarder(session: Session) -> None:
    """把处理线程产生的消息按序发送给前端。"""
    while True:
        payload = await session.outbox.get()
        try:
            await session.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            # 客户端已断开：停止转发，处理线程的 _put 会静默丢弃后续消息
            return


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "sample_rate": settings.sample_rate,
        "vad_threshold_s": settings.vad_threshold_s,
        "rec_threshold_s": settings.rec_threshold_s,
        "merge_gap_s": settings.merge_gap_s,
        "cluster_merge_thr": settings.cluster_merge_thr,
        "meetings_dir": str(settings.meetings_dir),
        "speakers_dir": str(settings.speakers_dir),
        "hotwords_dir": str(settings.hotwords_dir),
        "min_speaker_audio_s": settings.min_speaker_audio_s,
        "spk_match_threshold": settings.spk_match_threshold,
        "models": models.loaded_state(),
        "active_sessions": len(sessions),
        "active_meetings": len(meeting_sessions),
    }


# ═══════════════════════════════════════════════════════════════════
# 会议记录管理
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/meetings")
async def list_meetings() -> dict:
    meetings = await asyncio.get_running_loop().run_in_executor(None, meeting_store.list_meetings)
    return {"meetings": meetings}


@app.get("/api/meetings/{meeting_id}")
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


@app.get("/api/meetings/{meeting_id}/segments")
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


@app.post("/api/meetings")
async def create_meeting(payload: Optional[dict] = None) -> dict:
    title = (payload or {}).get("title", "") or ""
    meta = await asyncio.get_running_loop().run_in_executor(None, meeting_store.create, None, title)
    return {"meeting": meta}


@app.post("/api/meetings/{meeting_id}/save")
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


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> dict:
    if not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    ok = await asyncio.get_running_loop().run_in_executor(None, meeting_store.delete, meeting_id)
    if not ok:
        raise HTTPException(status_code=404, detail="meeting not found")
    return {"ok": True}


@app.get("/api/meetings/{meeting_id}/export")
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


# ═══════════════════════════════════════════════════════════════════
# 说话人声纹库管理
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/speakers")
async def list_speakers() -> dict:
    speakers = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.list_speakers
    )
    return {"speakers": speakers}


@app.post("/api/speakers")
async def create_speaker(payload: Optional[dict] = None) -> dict:
    name = (payload or {}).get("name", "") or ""
    speaker = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.create, name
    )
    return {"speaker": speaker}


@app.patch("/api/speakers/{speaker_id}")
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


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str) -> dict:
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.delete_speaker, speaker_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="speaker not found")
    return {"ok": True}


@app.post("/api/speakers/{speaker_id}/vectors/record")
async def add_speaker_record(speaker_id: str, request: Request) -> dict:
    """现场录音：请求体为 16kHz 单声道 Int16 小端 PCM，后端提取声纹向量。"""
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    body = await request.body()
    if len(body) < 2:
        raise HTTPException(status_code=400, detail="没有收到音频数据")
    pcm = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    duration_s = len(pcm) / settings.sample_rate
    if duration_s < 0.2:
        raise HTTPException(status_code=400, detail="录音过短，无法提取声纹特征")
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, models.embed, pcm)
    if emb is None:
        raise HTTPException(status_code=400, detail="无法从该录音提取声纹特征")
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


@app.post("/api/speakers/{speaker_id}/vectors/meeting")
async def add_speaker_from_meeting(speaker_id: str, payload: Optional[dict] = None) -> dict:
    """从已保存会议中选一条或多条 segment，使用记录中的声纹特征向量。"""
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    meeting_id = (payload or {}).get("meeting_id") or ""
    segment_ids = [int(s) for s in ((payload or {}).get("segment_ids") or [])]
    if not meeting_id or not meeting_store.valid_id(meeting_id):
        raise HTTPException(status_code=400, detail="invalid meeting id")
    if not segment_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个片段")

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
        raise HTTPException(status_code=400, detail="所选片段没有可用的声纹向量")

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


@app.delete("/api/speakers/{speaker_id}/vectors/{vector_id}")
async def delete_speaker_vector(speaker_id: str, vector_id: str) -> dict:
    if not speaker_store.valid_id(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, speaker_store.delete_vector, speaker_id, vector_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="vector not found")
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════
# 热词库管理
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/hotwords")
async def list_hotwords() -> dict:
    libraries = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.list_hotwords
    )
    return {"hotwords": libraries}


@app.post("/api/hotwords")
async def create_hotword_library(payload: Optional[dict] = None) -> dict:
    name = (payload or {}).get("name", "") or ""
    words = (payload or {}).get("words") or []
    library = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.create, name, words
    )
    return {"hotword": library}


@app.patch("/api/hotwords/{hotword_id}")
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


@app.delete("/api/hotwords/{hotword_id}")
async def delete_hotword_library(hotword_id: str) -> dict:
    if not hotword_store.valid_id(hotword_id):
        raise HTTPException(status_code=400, detail="invalid hotword id")
    ok = await asyncio.get_running_loop().run_in_executor(
        None, hotword_store.delete, hotword_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="hotword library not found")
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════
# 实时音频 WebSocket
# ═══════════════════════════════════════════════════════════════════
@app.websocket("/ws/audio")
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
    await websocket.accept()

    vad_s = float(vad_threshold if vad_threshold and vad_threshold > 0 else settings.vad_threshold_s)
    rec_s = float(rec_threshold if rec_threshold and rec_threshold > 0 else settings.rec_threshold_s)
    vad_s = max(0.5, min(vad_s, 30.0))
    rec_s = max(0.2, min(rec_s, 30.0))

    loop = asyncio.get_running_loop()
    loaded_state: Optional[dict] = None
    meeting_title = title or ""
    if meeting_id:
        if not meeting_store.valid_id(meeting_id):
            await websocket.close(code=1008)
            return
        if meeting_store.exists(meeting_id):
            loaded_state = await loop.run_in_executor(None, meeting_store.load_state, meeting_id)
            meta = (loaded_state or {}).get("meta") or {}
            meeting_title = meta.get("title") or meeting_title
        else:
            # 新会议：先建一条记录，结束/保存时再写入音频与识别状态
            await loop.run_in_executor(None, meeting_store.create, meeting_id, title or "")
        # 同一会议已有活动会话时，先关掉旧的
        old = meeting_sessions.get(meeting_id)
        if old is not None:
            old.stop()
            meeting_sessions.pop(meeting_id, None)

    # 会议参与者：优先使用前端显式传入的选择，否则回退到会议已保存的设置
    if participants is not None:
        participant_ids = [p.strip() for p in participants.split(",") if p.strip()]
    else:
        participant_ids = list((loaded_state or {}).get("participant_ids") or [])
    participants_explicit = participants is not None
    # 热词库：优先使用前端显式传入的选择，否则回退到会议已保存的设置
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

    sid = session_id or uuid.uuid4().hex
    if sid in sessions:
        sessions[sid].stop()
        del sessions[sid]

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
    session.start(loop, state=loaded_state)

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
                    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    session.buffer.append(pcm)
                text = message.get("text")
                if text:
                    try:
                        ctrl = json.loads(text)
                    except json.JSONDecodeError:
                        ctrl = {}
                    if ctrl.get("type") == "stop":
                        # 前端主动收尾：处理完剩余音频后关闭
                        stop_requested = True
                        break
            elif kind == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        # 触发处理器收尾（自动保存 + 把窗口剩余语音也确认掉），等它推完再关闭连接
        session.stop()
        # join 放 executor，避免阻塞事件循环导致 WebSocket 心跳超时
        await asyncio.get_running_loop().run_in_executor(None, session.processor.join, 120)
        # 给转发器一点时间把收尾消息（最终 transcript / closed）发完
        drain_deadline = time.monotonic() + 5.0
        while not session.outbox.empty() and time.monotonic() < drain_deadline:
            await asyncio.sleep(0.05)
        forwarder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await forwarder
        sessions.pop(sid, None)
        if meeting_id and meeting_sessions.get(meeting_id) is session:
            meeting_sessions.pop(meeting_id, None)
        if stop_requested:
            try:
                await websocket.close()
            except Exception:
                pass


@app.on_event("startup")
async def _preload_models() -> None:
    """启动时提前加载模型，避免首个请求等待。"""
    await asyncio.get_running_loop().run_in_executor(None, models.preload)

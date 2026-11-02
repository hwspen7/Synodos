"""A single WebSocket connection and its recording processor."""

from __future__ import annotations
import asyncio
from typing import Optional
from fastapi import WebSocket
from audio.buffer import AudioBuffer
from core.config import Settings
from core.runtime import models, settings
from sessions.processor import SessionProcessor
from sessions.lifecycle import SessionLifecycle, SessionPhase
from storage.meetings import MeetingStore
from storage.speakers import SpeakerStore
from storage.hotwords import HotwordStore

class Session:
    """A WebSocket session with an audio buffer, processor thread, and outbox."""

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
        self.lifecycle = SessionLifecycle()
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
        self.lifecycle.transition(SessionPhase.STARTING)
        try:
            self.processor.set_loop(loop)
            if state is not None:
                self.processor.load_state(state)
            self.processor.load_participants()
            self.processor.load_hotwords()
            self.processor.rebuild_labels()
            self.processor.start()
        except Exception:
            self.lifecycle.transition(SessionPhase.FAILED, "startup_failed")
            raise
        self.lifecycle.transition(SessionPhase.RUNNING)

    def stop(self) -> None:
        self.lifecycle.stop()
        self.processor.stop()

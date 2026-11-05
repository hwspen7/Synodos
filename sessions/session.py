"""A single WebSocket connection and its recording processor."""

from __future__ import annotations
import asyncio
import os
from typing import Optional
from fastapi import WebSocket
from audio.buffer import AudioBuffer
from core.config import Settings
from core.runtime import models, settings
from sessions.processor import SessionProcessor
from sessions.stable_speaker_channels import StableSpeakerChannels
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
        auto_match_saved: bool = True,
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
            auto_match_saved=auto_match_saved,
            hotwords_explicit=hotwords_explicit,
            spk_threshold_explicit=spk_threshold_explicit,
        )
        backend = os.getenv("MEETING_INDEPENDENT_DIARIZATION", "off").strip().lower()
        if backend not in ("off", "nemotron3"):
            raise ValueError("MEETING_INDEPENDENT_DIARIZATION must be off or nemotron3")
        self.independent_diarization = None
        if backend == "nemotron3":
            from speech.nemotron_stream import NemotronRealtimeWorker

            def notify_acoustic_updates():
                self.processor._async_ready.set()
                self.buffer.wake()

            self.independent_diarization = NemotronRealtimeWorker(
                on_update=notify_acoustic_updates,
            )
        # The optional track is independent of ASR sentence timestamps and
        # never substitutes synthetic words for undecodable speech.
        self.processor.independent_diarization = self.independent_diarization

    def start(self, loop: asyncio.AbstractEventLoop, state: Optional[dict] = None) -> None:
        self.lifecycle.transition(SessionPhase.STARTING)
        try:
            self.processor.set_loop(loop)
            if state is not None:
                self.processor.load_state(state)
                if self.independent_diarization is not None:
                    # AOSC is valid within one continuous PCM connection.
                    # A reconnect starts a new anonymous speaker namespace
                    # rather than falsely reusing the previous cache's IDs.
                    self.processor._independent_pcm_origin_samples = (
                        self.buffer.start_offset()
                    )
                    self.processor._independent_id_offset = max(
                        (int(item.get("speaker_id", -1))
                         for item in state.get("segments", [])
                         if type(item.get("speaker_id")) is int
                         and item["speaker_id"] > 0),
                        default=0,
                    )
                    self.processor._nemotron_channels = StableSpeakerChannels(
                        offset=self.processor._independent_id_offset
                    )
            self.processor.load_participants()
            self.processor.load_hotwords()
            self.processor.rebuild_labels()
            self.processor.start()
            if self.independent_diarization is not None:
                self.independent_diarization.start()
        except Exception:
            self.lifecycle.transition(SessionPhase.FAILED, "startup_failed")
            raise
        self.lifecycle.transition(SessionPhase.RUNNING)

    def stop(self) -> None:
        self.lifecycle.stop()
        if self.independent_diarization is not None:
            self.independent_diarization.request_finish()
        self.processor.stop()

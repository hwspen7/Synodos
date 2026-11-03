"""Background processor for a single WebSocket meeting session.

Processing pipeline:
1. Run ASR on buffered audio after the processing window threshold.
   Paraformer performs recognition with VAD, punctuation, and sentence timestamps.
2. Confirm sentences ending sufficiently before the end of the audio window.
   Confirmed transcripts do not undergo repeated recognition.
   CAM++ extracts a single embedding for each confirmed segment.
3. Assign speaker identities:
   Match each embedding against selected participant voiceprints.
   Use the configured cosine similarity threshold for participant matching.
   Cluster the remaining segments using FunASR ClusterBackend.
   Assign unknown-speaker labels when no known participant matches.
   Maintain stable speaker IDs across clustering updates.
4. Send complete segment snapshots when the confirmed transcript changes.
   The frontend groups adjacent same-speaker segments into paragraphs.

Confirmed segments are removed from the rolling buffer to reduce recomputation.
Their text and voiceprint embeddings remain unchanged after confirmation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, wait
import logging
import threading
import time
from typing import Optional

import numpy as np

from audio.buffer import AudioBuffer
from speech.config import load_settings
from speech.streaming import PreviewTracker
from core.config import Settings
from storage.hotwords import HotwordStore
from inference.models import ModelManager
from storage.speakers import SpeakerStore
from storage.meetings import MeetingStore
from sessions.save_queue import SaveQueue
from observability.metrics import metrics
from phase07.telemetry import telemetry
from phase07.speakers import speaker_stats, find_distinct_participant
from speech.commit_guard import is_temporal_duplicate
from phase07.recovery import autosave_seconds
from sessions.outbox import enqueue_message
from sessions.translation_tasks import submit_translation
from sessions.speaker_tasks import submit_speaker_analysis
from sessions.speaker_fallback import (
    UNIDENTIFIED_CLUSTER, fallback_speaker_id, normalized_embedding,
)

logger = logging.getLogger(__name__)


class SessionProcessor(threading.Thread):
    def __init__(
        self,
        session_id: str,
        buffer: AudioBuffer,
        settings: Settings,
        models: ModelManager,
        outbox: "asyncio.Queue[dict]",
        stop_event: Optional[threading.Event] = None,
        meeting_store: Optional[MeetingStore] = None,
        meeting_id: Optional[str] = None,
        meeting_title: str = "",
        speaker_store: Optional[SpeakerStore] = None,
        participant_ids: Optional[list[str]] = None,
        hotword_store: Optional[HotwordStore] = None,
        hotword_ids: Optional[list[str]] = None,
        spk_threshold: Optional[float] = None,
        participants_explicit: bool = False,
        hotwords_explicit: bool = False,
        spk_threshold_explicit: bool = False,
    ):
        super().__init__(name=f"session-{session_id[:8]}", daemon=True)
        self.session_id = session_id
        self.buffer = buffer
        self.settings = settings
        self.models = models
        self.outbox = outbox
        # Handle transcripts and speaker assignments.
        # Handle transcripts and speaker assignments.
        self._stop_event = stop_event or threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        sr = settings.sample_rate
        # Handle transcripts and speaker assignments.
        self._process_threshold_s = settings.vad_threshold_s
        # Handle transcripts and speaker assignments.
        self._trusted_threshold_s = settings.rec_threshold_s
        # Handle transcripts and speaker assignments.
        self._final_discard_s = settings.final_discard_s
        self._speech_settings = load_settings()
        self.speech_language = self._speech_settings.language
        self.translate_target = self._speech_settings.translate_to
        self._preview_tracker = PreviewTracker()
        if self._speech_settings.engine != "legacy":
            import os
            # Avoid repeatedly committing fragments from tiny ASR windows.
            self._process_threshold_s = max(4.0, min(self._process_threshold_s,
                                                     self._speech_settings.preview_interval_s))
            self._trusted_threshold_s = max(settings.rec_threshold_s,
                                            self._speech_settings.confirmation_delay_s)
            commit_s = float(os.getenv("MEETING_MIN_COMMIT_CONTEXT_S", "7.0"))
            if not 4.0 <= commit_s <= 30.0:
                raise ValueError("MEETING_MIN_COMMIT_CONTEXT_S must be in [4, 30]")
            self._min_commit_context_s = max(self._process_threshold_s, commit_s)

        # Handle transcripts and speaker assignments.
        self.segments: list[dict] = []
        self._next_segment_id = 1

        # Handle transcripts and speaker assignments.
        self._speaker_map: dict[int, int] = {}
        self._speaker_seg_ids: dict[int, set[int]] = {}

        self._last_fingerprint: Optional[tuple] = None
        self._transcript_version = 0
        self._last_state_sent = 0.0
        self._last_inferred_absolute_end = -1
        self._last_reclustered_segment_id = 0

        # Handle transcripts and speaker assignments.
        self.meeting_store = meeting_store
        self.meeting_id = meeting_id
        self.meeting_title = meeting_title
        self._save_queue = SaveQueue()
        self._save_requested = self._save_queue.wake_event
        self._final_save_ok = False
        self._phase07_autosave_interval = autosave_seconds()
        self._phase07_last_autosave = time.monotonic()
        self._phase07_saved_segment_id = 0
        self._translation_tasks: dict[int, Future] = {}
        self._speaker_tasks: dict[int, Future] = {}
        self._async_ready = threading.Event()

        # Handle transcripts and speaker assignments.
        self.speaker_store = speaker_store
        self.participant_ids: list[str] = list(participant_ids or [])
        # Handle transcripts and speaker assignments.
        self.hotword_store: Optional[HotwordStore] = hotword_store
        self.hotword_ids: list[str] = list(hotword_ids or [])
        self._hotword_str: str = ""
        # Handle transcripts and speaker assignments.
        self._hotword_limit = 200
        self.spk_threshold: float = float(
            spk_threshold
            if spk_threshold is not None
            else settings.spk_match_threshold
        )
        self._participants: list[dict] = []
        self._participant_label_map: dict[str, int] = {}
        # Handle transcripts and speaker assignments.
        self._unknown_index: dict[int, int] = {}
        # Handle transcripts and speaker assignments.
        self._participants_explicit = participants_explicit
        self._hotwords_explicit = hotwords_explicit
        self._spk_threshold_explicit = spk_threshold_explicit

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def stop(self) -> None:
        self._stop_event.set()
        self.buffer.wake()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._drain_save_requests():
                    continue
                if not self.buffer.wait_until(
                    self._process_threshold_s, self._stop_event,
                    self._save_requested, self._async_ready,
                ):
                    break
                if self._stop_event.is_set():
                    break
                if self._async_ready.is_set():
                    self._async_ready.clear()
                    speaker_updated = self._collect_speakers()
                    self._collect_translations()
                    if speaker_updated:
                        self._recluster()
                        self._last_reclustered_segment_id = self._next_segment_id - 1
                    if self.segments:
                        self._rebuild_and_push()
                    continue
                if self._save_requested.is_set():
                    continue
                try:
                    changed = self._cycle(final=False)
                except Exception:
                    logger.exception("session %s cycle failed", self.session_id)
                    changed = False
                if self._drain_save_requests():
                    continue
                if not changed:
                    # Handle transcripts and speaker assignments.
                    current = self.buffer.seconds()
                    # Avoid repeated full-buffer model inference for each 0.5s of silence.
                    self.buffer.wait_until(current + 1.5, self._stop_event,
                                           self._save_requested, self._async_ready)
            # Handle transcripts and speaker assignments.
            self._finalize()
        except Exception:
            metrics.increment("processor_crashes")
            logger.exception("session %s processor crashed", self.session_id)
            self._put({"type": "error", "message": "backend processor crashed"})
        finally:
            self._save_queue.close(succeeded=self._final_save_ok)
            self._put({"type": "closed", "saved": self._final_save_ok})

    def _finalize(self) -> None:
        if self.buffer.total_samples() > 0:
            try:
                self._cycle(final=True)
            except Exception:
                logger.exception("session %s final transcription failed", self.session_id)
                self._put({"type": "error", "message": "Final transcription failed; confirmed text may be incomplete"})
        speaker_updated = self._collect_speakers(wait_seconds=5.0)
        if speaker_updated:
            try:
                self._recluster()
                self._last_reclustered_segment_id = self._next_segment_id - 1
            except Exception:
                logger.exception("Final speaker clustering failed")
                self._fallback_labels()
        self._collect_translations(wait_seconds=2.0)
        self._rebuild_and_push()
        if self._speaker_tasks:
            self._put({"type": "feature_warning", "feature": "diarization",
                       "message": "Speaker analysis unfinished; uncertain speakers were kept unidentified"})
            self._speaker_tasks.clear()
        if self._translation_tasks:
            self._put({"type": "feature_warning", "feature": "translation",
                       "message": "Some translations were not completed before meeting shutdown"})
            self._translation_tasks.clear()
        self._final_save_ok = self._do_save()
        if not self._final_save_ok and self.meeting_store and self.meeting_id:
            self._put({"type": "error", "message": "Unable to save the final meeting transcript"})
        self._save_queue.close(succeeded=self._final_save_ok)

    # ------------------------------------------------------------ persistence
    def request_save(self, timeout: float = 30.0) -> bool:
        """Wait for this request's write to complete successfully."""
        if not self.meeting_store or not self.meeting_id:
            return False
        if self.stopped or not self.is_alive():
            return False
        ticket = self._save_queue.submit()
        if ticket is None:
            return False
        self.buffer.wake()
        return ticket.completed.wait(timeout=timeout) and ticket.succeeded

    def _drain_save_requests(self) -> bool:
        tickets = self._save_queue.take()
        if not tickets:
            return False
        succeeded = self._do_save()
        self._save_queue.finish(tickets, succeeded)
        return True

    def _do_save(self) -> bool:
        if not self.meeting_store or not self.meeting_id:
            return False
        try:
            self.meeting_store.save_state(self.snapshot_state())
            self._phase07_last_autosave = time.monotonic()
            self._phase07_saved_segment_id = max(
                self._phase07_saved_segment_id, self._next_segment_id - 1
            )
            logger.info("meeting %s saved (%d segments)", self.meeting_id, len(self.segments))
            return True
        except Exception:
            logger.exception("save meeting %s failed", self.meeting_id)
            return False

    def snapshot_state(self) -> dict:
        """Audio transcription and speaker data handling."""
        return {
            "id": self.meeting_id,
            "title": self.meeting_title,
            "sample_rate": self.settings.sample_rate,
            "merge_gap_s": self.settings.merge_gap_s,
            "participant_ids": list(self.participant_ids),
            "hotword_ids": list(self.hotword_ids),
            "spk_threshold": float(self.spk_threshold),
            "participant_label_map": dict(self._participant_label_map),
            "unknown_index": dict(self._unknown_index),
            "segments": [
                {
                    "id": s["id"],
                    "start_sample": s["start_sample"],
                    "end_sample": s["end_sample"],
                    "text": s["text"],
                    "cluster": s["cluster"],
                    "speaker_id": s.get(
                        "speaker_id",
                        self._speaker_map.get(s["cluster"], s["cluster"]),
                    ),
                    "speaker_name": s.get("speaker_name", ""),
                    "speaker_uncertain": bool(s.get("speaker_uncertain", False)),
                    "emb": s["emb"],  # Preserve behavior.
                    "language": s.get("language", "und"),
                    "language_confidence": s.get("language_confidence"),
                    "timestamp_quality": s.get("timestamp_quality", "segment"),
                    "words": s.get("words", []),
                    "translation": s.get("translation", ""),
                "speaker_uncertain": bool(s.get("speaker_uncertain", False)),
                    "revision": s.get("revision", 0),
                    "corrections": s.get("corrections", []),
                }
                for s in self.segments
            ],
            "next_segment_id": self._next_segment_id,
            "speaker_map": dict(self._speaker_map),
        }

    def load_state(self, state: dict) -> None:
        """Audio transcription and speaker data handling."""
        self.segments = []
        for item in state.get("segments", []) or []:
            self.segments.append(
                {
                    "id": int(item["id"]),
                    "start_sample": int(item["start_sample"]),
                    "end_sample": int(item["end_sample"]),
                    "emb": normalized_embedding(item.get("emb")),
                    "cluster": int(item.get("cluster", -1)),
                    "speaker_id": int(item.get("speaker_id", item.get("cluster", -1))),
                    "speaker_name": item.get("speaker_name", ""),
                    "speaker_uncertain": bool(item.get("speaker_uncertain", False)),
                    "text": item.get("text", "") or "",
                    "language": item.get("language", "und"),
                    "language_confidence": item.get("language_confidence"),
                    "timestamp_quality": item.get("timestamp_quality", "segment"),
                    "words": item.get("words", []),
                    "translation": item.get("translation", ""),
                    "revision": int(item.get("revision", 0)),
                    "corrections": item.get("corrections", []),
                }
            )
        if self.segments:
            max_id = max(s["id"] for s in self.segments)
        else:
            max_id = 0
        self._next_segment_id = max(int(state.get("next_segment_id", max_id + 1)), max_id + 1)
        self._phase07_saved_segment_id = max_id
        self._speaker_map = {int(k): int(v) for k, v in (state.get("speaker_map") or {}).items()}
        self._speaker_seg_ids = {}
        if not self._participants_explicit:
            self.participant_ids = list(state.get("participant_ids") or [])
        if not self._hotwords_explicit:
            self.hotword_ids = list(state.get("hotword_ids") or [])
        if not self._spk_threshold_explicit:
            self.spk_threshold = float(state.get("spk_threshold", self.spk_threshold))
        self._participant_label_map = {
            str(k): int(v) for k, v in (state.get("participant_label_map") or {}).items()
        }
        self._unknown_index = {
            int(k): int(v) for k, v in (state.get("unknown_index") or {}).items()
        }

        # Handle transcripts and speaker assignments.
        offset = max((int(s["end_sample"]) for s in self.segments), default=0)
        self.buffer.set_start_offset(offset)

        self._last_fingerprint = None
        self.meeting_title = state.get("title") or self.meeting_title

    def load_participants(self) -> None:
        """Audio transcription and speaker data handling."""
        if not self.speaker_store or not self.participant_ids:
            self._participants = []
            return
        self._participants = self.speaker_store.get_for_matching(self.participant_ids)
        # Handle transcripts and speaker assignments.
        self.participant_ids = [p["id"] for p in self._participants]

    def load_hotwords(self) -> None:
        """Audio transcription and speaker data handling."""
        if not self.hotword_store or not self.hotword_ids:
            self._hotword_str = ""
            return
        words = self.hotword_store.get_words_for_meeting(
            self.hotword_ids, limit=self._hotword_limit
        )
        # Handle transcripts and speaker assignments.
        valid_ids = {lib["id"] for lib in self.hotword_store.list_hotwords()}
        self.hotword_ids = [hid for hid in self.hotword_ids if hid in valid_ids]
        self._hotword_str = " ".join(words)
        if len(words) >= self._hotword_limit:
            logger.warning(
                "[%s] hotwords capped at %d, rest ignored",
                self.session_id[:8],
                self._hotword_limit,
            )

    def rebuild_labels(self) -> None:
        """Audio transcription and speaker data handling."""
        if self.segments:
            self._recluster()
            self._rebuild_and_push()
        self._send_state()

    # ------------------------------------------------------------------ cycle
    def _cycle(self, final: bool = False) -> bool:
        sr = self.settings.sample_rate
        start_offset, audio = self.buffer.snapshot()
        buffer_s = len(audio) / sr

        if len(audio) < sr * 0.2 and not final:
            self._send_state()
            return False
        absolute_end = start_offset + len(audio)
        # A rescan with no new microphone samples cannot improve recognition.
        # Finalization is exempt so the last incomplete utterance gets one chance.
        if not final and absolute_end == self._last_inferred_absolute_end:
            self._send_state()
            return False
        self._last_inferred_absolute_end = absolute_end

        # Handle transcripts and speaker assignments.
        phase07_asr_started = time.monotonic()
        if self._speech_settings.engine == "legacy":
            sentences = self.models.transcribe_sentences(audio, hotword=self._hotword_str or None)
        else:
            recognition = self.models.recognize_multilingual(
                audio,
                language=self.speech_language,
                terms=self._hotword_str,
                config=self._speech_settings,
            )
            sentences = [piece.as_legacy() for piece in recognition.utterances]
        telemetry.observe_asr_cycle(buffer_s, time.monotonic() - phase07_asr_started)
        unconfirmed = []
        local_end = 0
        for sentence in sentences:
            start_ms = int(sentence["start"])
            end_ms = int(sentence["end"])
            text = sentence["text"]
            # Handle transcripts and speaker assignments.
            should_hold = not final and (
                buffer_s - end_ms / 1000.0 < self._trusted_threshold_s
                or (
                    self._speech_settings.engine != "legacy"
                    and buffer_s < self._min_commit_context_s
                )
            )
            if should_hold:
                unconfirmed.append(sentence)
                continue

            s_g = start_offset + int(start_ms * sr / 1000)
            e_g = start_offset + int(end_ms * sr / 1000)
            if e_g <= s_g or start_ms < 0 or end_ms > buffer_s * 1000 + 250:
                continue
            if is_temporal_duplicate(s_g, e_g, text, self.segments):
                metrics.increment("asr_temporal_duplicate_skipped")
                local_end = max(local_end, e_g - start_offset)
                continue

            seg_audio = audio[s_g - start_offset : e_g - start_offset]
            from speech.turn_split import split_by_acoustic_turns
            pieces = split_by_acoustic_turns(
                sentence, seg_audio, self.models.embed, sample_rate=sr,
            ) if self._speech_settings.engine != "legacy" else [
                {"start": start_ms, "end": end_ms, "text": text,
                 "emb": None, "ambiguous": False}
            ]
            for piece in pieces:
                a = start_offset + int(piece["start"] * sr / 1000)
                b = start_offset + int(piece["end"] * sr / 1000)
                if b <= a or is_temporal_duplicate(a, b, piece["text"], self.segments):
                    continue
                emb = normalized_embedding(piece.get("emb"))
                if emb is None and not piece.get("ambiguous") and b - a >= int(sr * 1.2):
                    try:
                        emb = normalized_embedding(self.models.embed(audio[a-start_offset:b-start_offset]))
                    except Exception:
                        logger.exception("speaker embedding failed; retaining transcript")
                self.segments.append({
                    "id": self._next_segment_id,
                    "start_sample": a,
                    "end_sample": b,
                    "text": piece["text"],
                    "language": sentence.get("language", "und"),
                    "language_confidence": sentence.get("language_confidence"),
                    "timestamp_quality": sentence.get("timestamp_quality", "segment"),
                    "words": sentence.get("words", []),
                    "emb": emb,
                    "cluster": -1,
                    "speaker_uncertain": bool(piece.get("ambiguous")),
                })
                self._next_segment_id += 1
                if self.translate_target:
                    detected = str(sentence.get("language", self.speech_language)).lower()
                    if detected == self.translate_target:
                        self.segments[-1]["translation"] = piece["text"]
                    else:
                        task = submit_translation(piece["text"], detected, self.translate_target)
                        if task is None:
                            self._put({"type": "feature_warning", "feature": "translation",
                                       "message": "Translation backlog limit reached; source text preserved"})
                        else:
                            self._translation_tasks[self._next_segment_id - 1] = task
                logger.info(
                    "[%s] confirmed segment %d (%.2f ~ %.2f seconds)",
                    self.session_id[:8], self._next_segment_id - 1, a / sr, b / sr,
                )
                metrics.increment("asr_segments_confirmed")
            local_end = max(local_end, e_g - start_offset)

        if self._speech_settings.engine != "legacy" and not self._speech_settings.native_streaming and not final:
            preview_text = " ".join(str(s.get("text", "")) for s in unconfirmed).strip()
            if preview_text:
                from speech.streaming import PreviewTracker
                latest = unconfirmed[-1]
                event = self._preview_tracker.update(preview_text, latest.get("language", "und"),
                                                       start_offset * 1000 // sr,
                                                       (start_offset + len(audio)) * 1000 // sr)
                if event is not None:
                    self._put({"type": "partial", **vars(event)})
            elif local_end > 0:
                self._preview_tracker.clear()
                self._put({"type": "partial", "text": "", "language": "und", "revision": 0})
        if final and (self._speech_settings.enable_events or self._speech_settings.enable_overlap):
            try:
                from speech.analysis import analyze
                self._put({"type": "audio_analysis", "result": analyze(audio, self._speech_settings)})
            except Exception as exc:
                self._put({"type": "feature_warning", "feature": "audio_analysis", "message": str(exc)})
        # Handle transcripts and speaker assignments.
        if local_end > 0:
            # Preserve any earlier unfinished utterance before advancing audio.
            if unconfirmed:
                first_held_ms = min(max(0, int(item["start"])) for item in unconfirmed)
                local_end = min(local_end, first_held_ms * sr // 1000)
            if local_end > 0:
                self.buffer.drop_before(local_end)
        elif not final and not sentences and self._speech_settings.engine != "legacy" and recognition.speech_detected is False:
            # Reclaim only very-low-energy LEADING silence when ASR returned
            # no utterances. Preserve all uncertain or voice-like samples.
            import os
            if os.getenv("MEETING_SAFE_SILENCE_RECLAIM", "1").strip().lower() in ("1", "true", "yes"):
                from speech.silence_retention import safe_silence_prefix
                drop = safe_silence_prefix(audio, sr)
                if drop:
                    self.buffer.drop_before(drop)
                    metrics.increment("audio_safe_silence_reclaimed")

        self._collect_translations()
        changed = False
        if self.segments:
            try:
                if self._last_reclustered_segment_id != self._next_segment_id - 1:
                    self._recluster()
                    self._last_reclustered_segment_id = self._next_segment_id - 1
            except Exception:
                logger.exception("speaker clustering failed; publishing confirmed text")
                self._fallback_labels()
            changed = self._rebuild_and_push()

        # Autosave only when a new confirmed segment exists; never persist partial ASR.
        if (changed and self._phase07_autosave_interval and self.meeting_store and self.meeting_id
                and self._next_segment_id - 1 > self._phase07_saved_segment_id
                and time.monotonic() - self._phase07_last_autosave >= self._phase07_autosave_interval):
            if self._do_save():
                metrics.increment("phase07_autosaves")
            else:
                metrics.increment("phase07_autosave_errors")

        if not final:
            self._send_state()
        return changed

    def _collect_translations(self, wait_seconds: float = 0.0) -> None:
        if not self._translation_tasks:
            return
        if wait_seconds > 0:
            wait(tuple(self._translation_tasks.values()), timeout=wait_seconds)
        completed = [(sid, future) for sid, future in self._translation_tasks.items()
                     if future.done()]
        for sid, future in completed:
            self._translation_tasks.pop(sid, None)
            try:
                value = future.result()
            except Exception as exc:
                self._put({"type": "feature_warning", "feature": "translation",
                           "message": str(exc)})
                continue
            for segment in self.segments:
                if segment["id"] == sid:
                    segment["translation"] = value
                    break

    # --------------------------------------------------------------- cluster
    def _recluster(self) -> None:
        import os
        # Explicit single-speaker mode is intended for one-person test meetings.
        if os.getenv("MEETING_DIARIZATION_MODE", "cluster") == "single":
            for record in self.segments:
                record["cluster"] = 1
                record["speaker_id"] = 1
                record["speaker_name"] = "Speaker 1"
            return
        if os.getenv("MEETING_DIARIZATION_MODE", "cluster") == "centroid":
            from speech.diarization import SpeakerCentroidTracker
            tracker = SpeakerCentroidTracker(threshold=float(os.getenv("MEETING_SPEAKER_SIMILARITY", "0.72")))
            for record in self.segments:
                stable = record.get("speaker_id")
                if isinstance(stable, int) and stable > 0 and record.get("emb") is not None:
                    tracker.seed(stable, record["emb"])
            for record in self.segments:
                emb = record.get("emb")
                if emb is None:
                    record.setdefault("speaker_id", -1)
                    record.setdefault("speaker_name", "Unidentified Speaker")
                    continue
                pid, similarity = self._match_participant(emb)
                if pid is not None and similarity >= self.spk_threshold:
                    if pid not in self._participant_label_map:
                        self._participant_label_map[pid] = self._next_free_speaker_id()
                    record["cluster"] = -1
                    record["speaker_id"] = self._participant_label_map[pid]
                    record["speaker_name"] = next((p["name"] for p in self._participants if p["id"] == pid), "Participant")
                elif not isinstance(record.get("speaker_id"), int) or record["speaker_id"] <= 0:
                    stable_id = tracker.assign(emb)
                    record["cluster"] = -2
                    record["speaker_id"] = stable_id if stable_id is not None else -1
                    record["speaker_name"] = f"Speaker {stable_id}" if stable_id else "Unidentified Speaker"
            return
        """Recompute speaker assignments for all confirmed segments.

        Match segments to participants when cosine similarity is high enough.
        Cluster unmatched segments with the FunASR clustering backend.
        Map cluster labels to stable speaker IDs.
        """
        if not self.segments:
            return
        matched: dict[int, str] = {}
        pool: list[int] = []
        for i, seg in enumerate(self.segments):
            if seg["emb"] is None:
                seg["cluster"] = UNIDENTIFIED_CLUSTER
                continue
            participant_id, sim = self._match_participant(seg["emb"])
            if participant_id is not None and sim >= self.spk_threshold:
                matched[seg["id"]] = participant_id
                seg["cluster"] = -1
            else:
                seg["cluster"] = -2
                pool.append(i)

        # Handle transcripts and speaker assignments.
        if pool:
            embs = np.stack([self.segments[i]["emb"] for i in pool]).astype(np.float32)
            labels = self.models.cluster(embs)
            for idx, label in zip(pool, labels):
                self.segments[idx]["cluster"] = int(label)

        self._remap_speakers()
        self._assign_labels(matched)

    def _fallback_labels(self) -> None:
        """Publish all confirmed text even if clustering fails."""
        for seg in self.segments:
            if not isinstance(seg.get("speaker_id"), int):
                seg["speaker_id"] = fallback_speaker_id(seg["id"])
                seg["speaker_name"] = "Unidentified Speaker"

    def _match_participant(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """Audio transcription and speaker data handling."""
        if not self._participants:
            return None, -1.0
        best_pid: Optional[str] = None
        best_sim = -1.0
        for p in self._participants:
            for vec in p["vectors"]:
                sim = float(np.dot(emb, vec["emb"]))
                if sim > best_sim:
                    best_sim = sim
                    best_pid = p["id"]
        # Opt-in ambiguity guard across distinct enrolled participants.
        import os
        margin = float(os.getenv("MEETING_PHASE07_SPEAKER_MARGIN", "0"))
        if margin > 0 and self._participants:
            return find_distinct_participant(self._participants, emb, margin)
        return best_pid, best_sim

    def _next_free_speaker_id(self, used: Optional[set[int]] = None) -> int:
        if used is None:
            used = set(self._participant_label_map.values()) | set(self._speaker_map.values())
        free_id = 1
        while free_id in used:
            free_id += 1
        return free_id

    def _remap_speakers(self) -> None:
        """Audio transcription and speaker data handling."""
        clustered = [s for s in self.segments if s["cluster"] >= 0]
        raw_labels = sorted({s["cluster"] for s in clustered})
        old_map = self._speaker_map
        old_seg_ids = self._speaker_seg_ids
        used: set[int] = set(self._participant_label_map.values())
        new_map: dict[int, int] = {}
        new_seg_ids: dict[int, set[int]] = {}
        for raw in raw_labels:
            seg_ids = {s["id"] for s in clustered if s["cluster"] == raw}
            new_seg_ids[raw] = seg_ids
            best_stable: Optional[int] = None
            best_n = 0
            for old_raw, stable in old_map.items():
                if stable in used:
                    continue
                n = len(seg_ids & old_seg_ids.get(old_raw, set()))
                if n > best_n:
                    best_n = n
                    best_stable = stable
            if best_stable is not None and best_n >= 1:
                new_map[raw] = best_stable
                used.add(best_stable)
            else:
                # Handle transcripts and speaker assignments.
                free_id = self._next_free_speaker_id(used)
                new_map[raw] = free_id
                used.add(free_id)
        self._speaker_map = new_map
        self._speaker_seg_ids = new_seg_ids

    def _assign_labels(self, matched: dict[int, str]) -> None:
        """Audio transcription and speaker data handling."""
        for participant_id in matched.values():
            if participant_id not in self._participant_label_map:
                self._participant_label_map[participant_id] = self._next_free_speaker_id()
        names = {p["id"]: p["name"] for p in self._participants}
        for seg in self.segments:
            if seg["cluster"] == UNIDENTIFIED_CLUSTER:
                seg["speaker_id"] = fallback_speaker_id(seg["id"])
                seg["speaker_name"] = "Unidentified Speaker"
                continue
            if seg["cluster"] == -1:
                participant_id = matched.get(seg["id"])
                stable = self._participant_label_map.get(participant_id or "", 1)
                seg["speaker_id"] = stable
                seg["speaker_name"] = names.get(participant_id or "", "Participant")
            else:
                stable = self._speaker_map.get(seg["cluster"], seg["cluster"])
                seg["speaker_id"] = stable
                if self.participant_ids:
                    if stable not in self._unknown_index:
                        self._unknown_index[stable] = (
                            max(self._unknown_index.values(), default=0) + 1
                        )
                    seg["speaker_name"] = f"Unknown Speaker {self._unknown_index[stable]}"
                else:
                    seg["speaker_name"] = f"Speaker {stable}"

    # ---------------------------------------------------------------- push
    def _rebuild_and_push(self) -> bool:
        """Audio transcription and speaker data handling."""
        ordered = sorted(self.segments, key=lambda s: s["start_sample"])
        items = [
            {
                "id": s["id"],
                "speaker_id": s.get(
                    "speaker_id", self._speaker_map.get(s["cluster"], s["cluster"])
                ),
                "speaker_name": s.get("speaker_name")
                or f"Speaker {s.get('speaker_id', self._speaker_map.get(s['cluster'], s['cluster']))}",
                "start_ms": round(s["start_sample"] * 1000 / self.settings.sample_rate),
                "end_ms": round(s["end_sample"] * 1000 / self.settings.sample_rate),
                "text": s["text"],
                "language": s.get("language", "und"),
                "language_confidence": s.get("language_confidence"),
                "timestamp_quality": s.get("timestamp_quality", "segment"),
                "translation": s.get("translation", ""),
                "revision": s.get("revision", 0),
            }
            for s in ordered
        ]
        phase07_speaker = speaker_stats(items)
        metrics.gauge("phase07_speaker_changes", phase07_speaker["speaker_switches"])
        metrics.gauge("phase07_unidentified_segments", phase07_speaker["unidentified_segments"])
        fingerprint = tuple(
            (i["id"], i["speaker_id"], i["speaker_name"], i["start_ms"], i["end_ms"], i["text"],
             i["language"], i.get("language_confidence"), i.get("translation"))
            for i in items
        )
        if fingerprint == self._last_fingerprint:
            return False
        self._last_fingerprint = fingerprint
        self._transcript_version += 1
        self._put(
            {
                "type": "transcript",
                "version": self._transcript_version,
                "segments": items,
            }
        )
        return True

    # ------------------------------------------------------------------ outbox
    def _put(self, payload: dict) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(enqueue_message, self.outbox, payload)
        except RuntimeError:
            logger.warning("session %s outbox is no longer available", self.session_id)

    def _send_state(self) -> None:
        now = time.time()
        if now - self._last_state_sent < 1.0:
            return
        self._last_state_sent = now
        self._put(
            {
                "type": "state",
                "buffered_ms": int(self.buffer.seconds() * 1000),
            }
        )

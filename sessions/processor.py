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
import logging
import threading
import time
from typing import Optional

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from storage.hotwords import HotwordStore
from inference.models import ModelManager
from storage.speakers import SpeakerStore
from storage.meetings import MeetingStore

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

        # Handle transcripts and speaker assignments.
        self.segments: list[dict] = []
        self._next_segment_id = 1

        # Handle transcripts and speaker assignments.
        self._speaker_map: dict[int, int] = {}
        self._speaker_seg_ids: dict[int, set[int]] = {}

        self._last_fingerprint: Optional[tuple] = None
        self._transcript_version = 0
        self._last_state_sent = 0.0

        # Handle transcripts and speaker assignments.
        self.meeting_store = meeting_store
        self.meeting_id = meeting_id
        self.meeting_title = meeting_title
        self._save_requested = threading.Event()
        self._save_done = threading.Event()

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
                    self._process_threshold_s, self._stop_event, self._save_requested
                ):
                    break
                if self._stop_event.is_set():
                    break
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
                    self.buffer.wait_until(current + 0.5, self._stop_event, self._save_requested)
            # Handle transcripts and speaker assignments.
            self._finalize()
        except Exception:
            logger.exception("session %s processor crashed", self.session_id)
            self._put({"type": "error", "message": "backend processor crashed"})
        finally:
            self._save_requested.clear()
            self._save_done.set()
            self._put({"type": "closed"})

    def _finalize(self) -> None:
        if self.buffer.total_samples() <= 0 and not self.segments:
            self._save_done.set()
            return
        if self.buffer.seconds() >= self._final_discard_s:
            try:
                self._cycle(final=True)
            except Exception:
                logger.exception("session %s final cycle failed", self.session_id)
        # Handle transcripts and speaker assignments.
        self._do_save()
        self._save_done.set()

    # ------------------------------------------------------------ persistence
    def request_save(self, timeout: float = 30.0) -> bool:
        """Audio transcription and speaker data handling."""
        if not self.meeting_store or not self.meeting_id:
            return False
        self._save_requested.set()
        self.buffer.wake()
        return self._save_done.wait(timeout=timeout)

    def _drain_save_requests(self) -> bool:
        if not self._save_requested.is_set():
            return False
        try:
            self._do_save()
        finally:
            self._save_requested.clear()
            self._save_done.set()
        return True

    def _do_save(self) -> None:
        if not self.meeting_store or not self.meeting_id:
            return
        try:
            self.meeting_store.save_state(self.snapshot_state())
            logger.info("meeting %s saved (%d segments)", self.meeting_id, len(self.segments))
        except Exception:
            logger.exception("save meeting %s failed", self.meeting_id)

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
                    "emb": s["emb"],  # Preserve behavior.
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
                    "emb": np.asarray(item["emb"], dtype=np.float32),
                    "cluster": int(item.get("cluster", -1)),
                    "speaker_id": int(item.get("speaker_id", item.get("cluster", -1))),
                    "speaker_name": item.get("speaker_name", ""),
                    "text": item.get("text", "") or "",
                }
            )
        if self.segments:
            max_id = max(s["id"] for s in self.segments)
        else:
            max_id = 0
        self._next_segment_id = max(int(state.get("next_segment_id", max_id + 1)), max_id + 1)
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

        if len(audio) < sr * 0.2:
            if not final:
                self._send_state()
            return False

        # Handle transcripts and speaker assignments.
        sentences = self.models.transcribe_sentences(audio, hotword=self._hotword_str or None)
        local_end = 0
        for sentence in sentences:
            start_ms = int(sentence["start"])
            end_ms = int(sentence["end"])
            text = sentence["text"]
            # Handle transcripts and speaker assignments.
            if not final and buffer_s - end_ms / 1000.0 < self._trusted_threshold_s:
                break

            s_g = start_offset + int(start_ms * sr / 1000)
            e_g = start_offset + int(end_ms * sr / 1000)
            if e_g <= s_g:
                continue

            # Handle transcripts and speaker assignments.
            seg_audio = audio[s_g - start_offset : e_g - start_offset]
            emb = self.models.embed(seg_audio)
            if emb is None:
                continue

            self.segments.append(
                {
                    "id": self._next_segment_id,
                    "start_sample": s_g,
                    "end_sample": e_g,
                    "text": text,
                    "emb": emb,
                    "cluster": -1,
                }
            )
            self._next_segment_id += 1
            local_end = max(local_end, e_g - start_offset)
            logger.info(
                "[%s] [%.2f ~ %.2f] %s",
                self.session_id[:8],
                s_g / sr,
                e_g / sr,
                text,
            )

        # Handle transcripts and speaker assignments.
        if local_end > 0:
            self.buffer.drop_before(local_end)

        changed = False
        if self.segments:
            self._recluster()
            changed = self._rebuild_and_push()

        if not final:
            self._send_state()
        return changed

    # --------------------------------------------------------------- cluster
    def _recluster(self) -> None:
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
                or f"Speaker {s.get('speaker_id', self._speaker_map.get(s['cluster'], s['cluster']))}"
                "start_ms": round(s["start_sample"] * 1000 / self.settings.sample_rate),
                "end_ms": round(s["end_sample"] * 1000 / self.settings.sample_rate),
                "text": s["text"],
            }
            for s in ordered
        ]
        fingerprint = tuple(
            (i["id"], i["speaker_id"], i["speaker_name"], i["start_ms"], i["end_ms"], i["text"])
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
            asyncio.run_coroutine_threadsafe(self.outbox.put(payload), loop)
        except Exception:
            pass

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

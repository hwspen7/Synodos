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
import os
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
from speech.commit_guard import is_temporal_duplicate, is_adjacent_tail_echo
from speech.commit_boundary import completed_island_drop, unclosed_voice_turn
from speech.acoustic_guard import uninformative_audio
from sessions.speaker_birth_guard import tentative_new_clusters
from speech.coverage_retention import choose_safe_drop
from speech.quality_profile import (
    quality_candidate_enabled, speaker_candidate_enabled,
)
from speech.nemotron_activity import (
    activity_turns, choose_caption_speaker, choose_caption_speaker_from_turns,
)
from speech.caption_acoustic_alignment import caption_support_summary
from speech.content_first_projection import project_confirmed_caption
from speech.asr_stage_trace import shifted
from speech.contracts import Utterance
from speech.language import normalize_language
from speech.supported_languages import PHASE1_LANGUAGE_SET
from phase07.recovery import autosave_seconds
from sessions.outbox import enqueue_message
from sessions.translation_tasks import submit_translation
from sessions.speaker_tasks import submit_speaker_analysis, submit_acoustic_window
from sessions.speaker_evidence import speaker_evidence
from sessions.registered_identity import match_saved_voice
from sessions.corroborated_speaker_channels import CorroboratedSpeakerChannels
from sessions.speaker_cluster_birth import gate_new_cluster_births
from sessions.short_turn_identity import recover_short_turn_identity
from sessions.fused_short_turn_recovery import recover_fused_short_turns
from sessions.retroactive_channel_recovery import (
    backfill_confirmed_channels, reconcile_final_channels,
)
from sessions.speaker_partition_evidence import consistent_partition_embedding
from sessions.speaker_resolution import preserve_supported_camplus
from speech.acoustic_timeline import AcousticTimeline
from speech.diarization import resolve_diarization_mode
from speech.turn_integrity import valid_turn_partition
from sessions.speaker_fallback import (
    UNIDENTIFIED_CLUSTER, fallback_speaker_id, normalized_embedding,
)
from sessions.speaker_stability import (
    collect_mature_labels, restore_mature_labels, stabilized_cluster_ids,
    summarize_cluster_fragmentation, diagnose_cluster_merge_thresholds,
    diagnose_fresh_cluster_candidates,
    anonymous_speaker_timeline,
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
        auto_match_saved: bool = True,
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
            # Avoid repeatedly committing fragments from tiny ASR windows.
            # Render a provisional caption after the configured preview interval.
            # Confirmed commits still require the longer acoustic context below.
            self._process_threshold_s = max(1.5, min(self._process_threshold_s,
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
        # New anonymous speakers need repeat acoustic evidence; known IDs
        # remain stable across incremental clustering and meeting resumes.
        self._speaker_birth_approved: set[int] = set()
        self._speaker_seg_ids: dict[int, set[int]] = {}
        # Limit retroactive diarization relabeling in long live meetings.
        # Set to 0 to retain unlimited historical reclustering for A/B tests.
        self._speaker_relabel_horizon_s = float(
            os.getenv("MEETING_SPEAKER_LABEL_REVISABLE_S", "120"))
        if not 0 <= self._speaker_relabel_horizon_s <= 3600:
            raise ValueError("MEETING_SPEAKER_LABEL_REVISABLE_S must be 0..3600")

        self._last_fingerprint: Optional[tuple] = None
        # Published Nemotron/CAM++ projection lives separately from raw
        # CAM++ acoustic embeddings and reclustering votes.
        self.independent_diarization = None
        self._independent_published: dict[int, int] = {}
        self._independent_scores: dict[int, dict] = {}
        self._independent_support: dict[int, dict] = {}
        self._fused_final_status: dict[int, dict] = {}
        self._content_first_projection: dict[int, dict] = {}
        self._independent_pcm_origin_samples = 0
        self._independent_id_offset = 0
        self._nemotron_channels = CorroboratedSpeakerChannels()
        self._transcript_version = 0
        self._last_state_sent = 0.0
        self._last_inferred_absolute_end = -1
        self._asr_failure_reported = False
        # Acoustic speech without decoded text is never silently considered
        # complete. A bounded retry is followed by an explicit gap record.
        self._coverage_hold_attempts: dict[int, int] = {}
        self._speech_coverage_gaps: list[dict] = []
        self._asr_stage_trace: list[dict] = []
        self._last_reclustered_segment_id = 0
        self._last_clustering_receipt: dict = {
            "strategy_executed": "not_run",
            "fallback_reason": None,
        }
        self._speaker_min_audio_s = float(
            os.getenv(
                "MEETING_SPEAKER_PROBE_MIN_S",
                "0.9" if speaker_candidate_enabled() else "1.2",
            )
        )
        if not 0.8 <= self._speaker_min_audio_s <= 1.2:
            raise ValueError("MEETING_SPEAKER_PROBE_MIN_S must be 0.8..1.2")

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
        self.ui_settings: dict = {}
        self._translation_tasks: dict[int, Future] = {}
        self._speaker_tasks: dict[int, Future] = {}
        # A saturated global speaker queue must not silently discard the
        # voiceprint for a confirmed ASR segment. Keep only a small bounded
        # set of PCM clips, retrying when an in-flight worker releases a slot.
        # The canonical caption is never changed by queue management.
        self._speaker_deferred: dict[int, tuple[dict, np.ndarray]] = {}
        self._speaker_deferred_limit = 8
        self._speaker_deferred_max_samples = 30 * sr
        self._speaker_backpressure_warned = False
        # Optional ECAPA shadow samples stay in memory and never enter saved state.
        self._ecapa_shadow = None
        self._ecapa_shadow_audio: dict[int, np.ndarray] = {}
        self._ecapa_shadow_sample_count = 0
        self._ecapa_shadow_last_sample = -10**12
        self._ecapa_shadow_seen_labels: set[int] = set()
        self._acoustic_turn_splits = 0
        self._acoustic_timeline = AcousticTimeline(sr)
        self._acoustic_tasks: dict[int, Future] = {}
        self._next_acoustic_sample = 0
        self._acoustic_window_samples = int(2.0 * sr)
        self._acoustic_stride_samples = int(1.5 * sr)
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
        self.active_hotword_count = 0
        self.spk_threshold: float = float(
            spk_threshold
            if spk_threshold is not None
            else settings.spk_match_threshold
        )
        self._participants: list[dict] = []
        self._auto_profile_mode = False
        self._auto_match_cache: dict[int, tuple[Optional[str], float]] = {}
        self.auto_match_saved = bool(auto_match_saved)
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
            # Lazily admit one optional background ECAPA session. This does
            # not load its model and cannot affect the live CAM++ path.
            try:
                from sessions.speaker_shadow import create_shadow_session
                self._ecapa_shadow = create_shadow_session()
            except Exception:
                logger.warning("Optional ECAPA shadow unavailable; CAM++ unchanged")
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
                    # Completed inference is independent of whether ASR has
                    # new PCM. Collect it even above the ASR buffer threshold.
                    below_threshold = self.buffer.seconds() < self._process_threshold_s
                    if (below_threshold or self.buffer.absolute_end_samples()
                            == self._last_inferred_absolute_end):
                        try:
                            self._refresh_auxiliary_results()
                        except Exception:
                            logger.exception("Unable to refresh asynchronous meeting evidence")
                        if below_threshold:
                            continue
                if self._save_requested.is_set():
                    continue
                try:
                    changed = self._cycle(final=False)
                except Exception as exc:
                    logger.exception("session %s cycle failed", self.session_id)
                    if not self._asr_failure_reported:
                        self._asr_failure_reported = True
                        self._put({
                            "type": "error",
                            "message": (
                                f"Speech recognition failed ({type(exc).__name__}). "
                                "Inspect backend exception_message and exception_traceback."
                            ),
                        })
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
            closed_payload = {"type": "closed", "saved": self._final_save_ok}
            if self._ecapa_shadow is not None:
                try:
                    # Final transcript is already saved. Wait at most 1.5s for
                    # pending diagnostics, then account for any discarded work.
                    closed_payload["speaker_shadow_diagnostics"] = (
                        self._ecapa_shadow.finish(wait_seconds=1.5)
                    )
                except Exception:
                    logger.warning("Optional ECAPA shadow diagnostics unavailable")
                finally:
                    self._ecapa_shadow.close()
            self._ecapa_shadow_audio.clear()
            independent = (
            None if resolve_diarization_mode(os.getenv("MEETING_DIARIZATION_MODE")) == "single"
            else getattr(self, "independent_diarization", None)
        )
            if independent is not None:
                independent_report = independent.report()
                closed_payload["independent_diarization"] = independent_report
                if independent_report.get("state") != "completed":
                    self._put({
                        "type": "feature_warning",
                        "feature": "diarization",
                        "message": (
                            "Independent Nemotron diarization did not finish "
                            "successfully. Automatic speaker assignments "
                            "require acoustic review; this meeting must not "
                            "be counted as a passed multi-speaker test."
                        ),
                    })
                try:
                    from speech.overlap_content_risk import overlap_content_risk
                    official = independent.final_turns_snapshot()
                    if official is not None:
                        sr = self.settings.sample_rate
                        caption_activity = []
                        for seg in self.segments:
                            if seg.get("speaker_override"):
                                sid = int(seg.get("speaker_override_id") or -1)
                            else:
                                sid = self._independent_published.get(
                                    int(seg["id"]), int(seg.get("speaker_id", -1))
                                )
                            caption_activity.append({
                                "start_ms": round(seg["start_sample"] * 1000 / sr),
                                "end_ms": round(seg["end_sample"] * 1000 / sr),
                                "speaker_id": sid,
                            })
                        origin_ms = round(
                            self._independent_pcm_origin_samples * 1000 / sr
                        )
                        original_audit = overlap_content_risk(
                            official, caption_activity,
                            acoustic_origin_ms=origin_ms,
                            speaker_id_offset=self._independent_id_offset,
                            speaker_id_map=getattr(
                                self, "_nemotron_channels", CorroboratedSpeakerChannels()
                            ).labels,
                        )
                        # Derive the actual display-level voice hypothesis
                        # independently, without counting a split caption's
                        # parent as an additional simultaneous speaker.
                        from speech.content_first_projection import (
                            project_caption_activity_for_audit,
                        )
                        projection = project_caption_activity_for_audit(
                            [
                                {
                                    "id": int(seg["id"]),
                                    "start_ms": round(seg["start_sample"] * 1000 / sr),
                                    "end_ms": round(seg["end_sample"] * 1000 / sr),
                                    "speaker_id": row["speaker_id"],
                                }
                                for seg, row in zip(
                                    self.segments, caption_activity
                                )
                            ],
                            self._content_first_projection,
                        )
                        original_audit["actual_ui_display_projection"] = (
                            overlap_content_risk(
                                official, projection,
                                acoustic_origin_ms=origin_ms,
                                speaker_id_offset=self._independent_id_offset,
                                speaker_id_map=getattr(
                                    self, "_nemotron_channels", CorroboratedSpeakerChannels()
                                ).labels,
                            )
                        )
                        original_audit["projection_note"] = (
                            "UI speaker cues replace their parent captions "
                            "for this separate activity diagnostic. It does "
                            "not establish second-speaker word recovery, "
                            "source-separated ASR, or improved canonical DER."
                        )
                        closed_payload["content_overlap_audit"] = original_audit
                except (AttributeError, TypeError, ValueError):
                    logger.warning(
                        "Optional final overlapping-voice audit unavailable",
                        exc_info=True,
                    )
            if self._asr_stage_trace:
                closed_payload["asr_coverage_trace"] = {
                    "schema": "meeting-asr-stage-trace-v1",
                    "cycles": list(self._asr_stage_trace),
                    "truncated": len(self._asr_stage_trace) >= 1024,
                    "contains_transcript_text": False,
                }
            if os.getenv("MEETING_DIARIZATION_DIAGNOSTICS", "0").lower() in (
                "1", "true", "yes"
            ):
                # End-of-session diagnostic only; never let profiling prevent
                # closing an otherwise successfully processed meeting.
                try:
                    summary = summarize_cluster_fragmentation(self.segments)
                    summary["acoustic_turn_splits"] = self._acoustic_turn_splits
                    summary["configured_merge_threshold"] = (
                        self.settings.cluster_merge_thr
                    )
                    summary["posthoc_merge_sweep"] = (
                        diagnose_cluster_merge_thresholds(self.segments)
                    )
                    summary["fresh_cluster_sweep"] = (
                        diagnose_fresh_cluster_candidates(self.segments)
                    )
                    # This is the requested mode. The bounded candidate
                    # may fall back to FunASR on excessive fragmentation.
                    summary["clustering_strategy_requested"] = os.getenv(
                        "MEETING_SPEAKER_CLUSTER_STRATEGY",
                        "candidate" if speaker_candidate_enabled() else "bootstrap",
                    )
                    summary["clustering_execution"] = dict(
                        self._last_clustering_receipt
                    )
                    # Review speaker-turn alignment against the original
                    # recording without transmitting any recognized text.
                    timeline_records = []
                    for segment in self.segments:
                        item = dict(segment)
                        if segment["id"] in self._independent_published:
                            item["speaker_id"] = self._independent_published[
                                segment["id"]
                            ]
                        if segment["id"] in self._independent_support:
                            item.update(self._independent_support[segment["id"]])
                        if segment["id"] in self._content_first_projection:
                            p = self._content_first_projection[segment["id"]]
                            item["caption_cues"] = p["cues"]
                            item["overlap_intervals_ms"] = p[
                                "overlap_intervals_ms"
                            ]
                            item["caption_projection_status"] = p["status"]
                        timeline_records.append(item)
                    summary["anonymous_timeline"] = anonymous_speaker_timeline(
                        timeline_records, self.settings.sample_rate
                    )
                    closed_payload["diarization_diagnostics"] = summary
                except Exception:
                    logger.exception("Optional diarization diagnostics failed")
            self._put(closed_payload)

    def _finalize(self) -> None:
        if self.buffer.total_samples() > 0:
            try:
                self._cycle(final=True)
            except Exception:
                logger.exception("session %s final transcription failed", self.session_id)
                self._put({"type": "error", "message": "Final transcription failed; confirmed text may be incomplete"})
        # A completed speaker job can admit another deferred voiceprint.
        # Drain *all* admitted waves within a bounded shutdown budget rather
        # than making only two passes and abandoning the final wave.
        speaker_updated = self._drain_final_speaker_analysis()
        acoustic_updated = self._collect_acoustic(wait_seconds=6.0)
        # A saved final meeting must never claim an unfinished worker is
        # still pending. Keep the unassigned identity explicit and retain ASR.
        incomplete_speaker_ids = (
            set(self._speaker_tasks) | set(self._speaker_deferred)
        )
        for record in getattr(self, "segments", ()):
            if (record.get("id") in incomplete_speaker_ids
                    and record.get("emb") is None
                    and record.get("speaker_evidence_status") in (
                        "pending_turn_analysis", "deferred_turn_analysis")):
                record["speaker_evidence_status"] = "turn_analysis_unfinished"
                record["speaker_uncertain"] = True
        if speaker_updated or acoustic_updated or speaker_candidate_enabled():
            try:
                self._recluster(
                    allow_historical_relabel=speaker_candidate_enabled(),
                )
                self._last_reclustered_segment_id = self._next_segment_id - 1
            except Exception:
                logger.exception("Final speaker clustering failed")
                self._fallback_labels()
        self._collect_translations(wait_seconds=2.0)
        # Finish independent acoustic inference BEFORE assigning/saving final
        # speaker labels. A request_finish() alone only signals the worker.
        independent = getattr(self, "independent_diarization", None)
        finish_independent = getattr(independent, "finish", None)
        if callable(finish_independent):
            try:
                completed = bool(finish_independent(timeout=20.0))
            except Exception:
                logger.exception("Independent diarization finalization failed")
                completed = False
            if not completed:
                self._put({
                    "type": "feature_warning",
                    "feature": "diarization",
                    "message": (
                        "Independent speaker finalization was incomplete; "
                        "unconfirmed identities remain provisional."
                    ),
                })
        self._rebuild_and_push()
        if self._speaker_tasks or self._acoustic_tasks or self._speaker_deferred:
            self._put({"type": "feature_warning", "feature": "diarization",
                       "message": "Speaker analysis unfinished; uncertain speakers were kept unidentified"})
            self._speaker_tasks.clear()
            self._acoustic_tasks.clear()
            self._speaker_deferred.clear()
        if self._translation_tasks:
            self._put({"type": "feature_warning", "feature": "translation",
                       "message": "Some translations were not completed before meeting shutdown"})
            self._translation_tasks.clear()
        self._final_save_ok = self._do_save()
        if not self._final_save_ok and self.meeting_store and self.meeting_id:
            self._put({"type": "error", "message": "Unable to save the final meeting transcript"})
        self._save_queue.close(succeeded=self._final_save_ok)

    def _drain_final_speaker_analysis(self, *, budget_seconds: float = 24.0) -> bool:
        """Finish successive waves of voiceprint jobs without unbounded shutdown.

        Each completed future can submit one or more deferred jobs. Draining
        only twice silently abandons the last wave in a fast conversation.
        The global ASR/voice worker pool is still bounded by its scheduler.
        """
        if not 0 < budget_seconds <= 120:
            raise ValueError("Invalid speaker finalization budget")
        deadline = time.monotonic() + budget_seconds
        updated = False
        while self._speaker_tasks or self._speaker_deferred:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            before = (len(self._speaker_tasks), len(self._speaker_deferred))
            updated |= self._collect_speakers(wait_seconds=min(2.0, remaining))
            after = (len(self._speaker_tasks), len(self._speaker_deferred))
            if (before == after and not self._speaker_tasks
                    and self._speaker_deferred):
                # Global worker slots may be busy with another meeting.
                # Avoid a CPU spin while retrying a bounded deferred queue.
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return updated

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

    def _autosave_if_due(self, changed: bool) -> None:
        """Periodically checkpoint confirmed text and revised speaker labels."""
        if (not changed or not self.segments
                or not self._phase07_autosave_interval
                or not self.meeting_store or not self.meeting_id
                or time.monotonic() - self._phase07_last_autosave
                   < self._phase07_autosave_interval):
            return
        if self._do_save():
            metrics.increment("phase07_autosaves")
        else:
            metrics.increment("phase07_autosave_errors")

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
        acoustics = speaker_evidence(self.segments)
        return {
            "id": self.meeting_id,
            "title": self.meeting_title,
            "sample_rate": self.settings.sample_rate,
            "merge_gap_s": self.settings.merge_gap_s,
            "participant_ids": list(self.participant_ids),
            "auto_match_saved": bool(self.auto_match_saved),
            "hotword_ids": list(self.hotword_ids),
            "spk_threshold": float(self.spk_threshold),
            "participant_label_map": dict(self._participant_label_map),
            "unknown_index": dict(self._unknown_index),
            "speech_coverage_gaps": list(self._speech_coverage_gaps),
            "segments": [
                {
                    "id": s["id"],
                    "start_sample": s["start_sample"],
                    "end_sample": s["end_sample"],
                    "text": s["text"],
                    "cluster": s["cluster"],
                    "speaker_id": (
                        int(s.get("speaker_override_id") or -1)
                        if s.get("speaker_override")
                        else s.get("speaker_id", -1)
                        if s.get("matched_profile_id") and s.get("matched_profile_name")
                        else self._independent_published.get(
                            int(s["id"]),
                            s.get("saved_nemotron_id") or s.get(
                                "speaker_id",
                                self._speaker_map.get(s["cluster"], s["cluster"]),
                            ),
                        )
                    ),
                    "speaker_name": (
                        ("Unidentified Speaker" if self._independent_published[
                            int(s["id"])
                        ] <= 0 else f"Speaker {self._independent_published[int(s['id'])]}")
                        if (int(s["id"]) in self._independent_published
                            and not s.get("speaker_override")
                            and not s.get("matched_profile_id"))
                        else s.get("speaker_name", "")
                    ),
                    "speaker_diarization_source": self._fused_final_status.get(
                        int(s["id"]), {},
                    ).get("speaker_diarization_source") or (
                        ("nemotron3" if self._independent_published[
                            int(s["id"])
                        ] > 0 else "nemotron3_uncertain")
                        if (int(s["id"]) in self._independent_published
                            and not s.get("speaker_override")
                            and not s.get("matched_profile_id"))
                        else ("manual_override" if s.get("speaker_override")
                              else "registered_voiceprint" if s.get("matched_profile_id")
                              else s.get("speaker_diarization_source", "camplus"))
                    ),
                    "speaker_acoustic_channel": self._independent_scores.get(
                        int(s["id"]), {},
                    ).get("speaker_acoustic_channel", s.get("speaker_acoustic_channel")),
                    "speaker_channel_gate_reason": self._independent_scores.get(
                        int(s["id"]), {},
                    ).get("speaker_channel_gate_reason", s.get("speaker_channel_gate_reason")),
                    "speaker_activity_score": self._independent_scores.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_score", s.get("speaker_activity_score")),
                    "speaker_activity_margin": self._independent_scores.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_margin", s.get("speaker_activity_margin")),
                    "speaker_activity_status": self._independent_scores.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_status", s.get("speaker_activity_status")),
                    "speaker_activity_intervals_ms": self._independent_support.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_intervals_ms", s.get(
                        "speaker_activity_intervals_ms", [],
                    )),
                    "speaker_activity_fraction": self._independent_support.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_fraction", s.get(
                        "speaker_activity_fraction",
                    )),
                    "speaker_activity_time_source": self._independent_support.get(
                        int(s["id"]), {},
                    ).get("speaker_activity_time_source", s.get(
                        "speaker_activity_time_source",
                    )),
                    "caption_cues": (
                        [] if s.get("speaker_override")
                        else self._content_first_projection.get(
                            int(s["id"]), {},
                        ).get("cues", s.get("caption_cues", []))
                    ),
                    "overlap_intervals_ms": self._content_first_projection.get(
                        int(s["id"]), {},
                    ).get("overlap_intervals_ms", s.get(
                        "overlap_intervals_ms", [],
                    )),
                    "caption_projection_status": (
                        "manual_override" if s.get("speaker_override")
                        else self._content_first_projection.get(
                            int(s["id"]), {},
                        ).get("status", s.get(
                            "caption_projection_status", "disabled",
                        ))
                    ),
                    "speaker_uncertain": bool(self._fused_final_status.get(
                        int(s["id"]), {},
                    ).get("speaker_uncertain", s.get("speaker_uncertain", False))),
                    "emb": s["emb"],  # Preserve behavior.
                    "language": s.get("language", "und"),
                    "language_confidence": s.get("language_confidence"),
                    "asr_confidence": s.get("asr_confidence"),
                    "recognition_origin": s.get("recognition_origin", "primary"),
                    "speaker_match_similarity": s.get("speaker_match_similarity"),
                    "matched_profile_id": s.get("matched_profile_id"),
                    "matched_profile_name": s.get("matched_profile_name"),
                    "speaker_evidence_status": s.get("speaker_evidence_status", "unavailable"),
                    "speaker_upstream_evidence_status": s.get(
                        "speaker_upstream_evidence_status",
                    ),
                    "speaker_birth_decision": s.get("speaker_birth_decision"),
                    **acoustics[s["id"]],
                    "speaker_status": self._fused_final_status.get(int(s["id"]), {}).get(
                        "speaker_status", acoustics[s["id"]]["speaker_status"]),
                    "speaker_uncertain": self._fused_final_status.get(int(s["id"]), {}).get(
                        "speaker_uncertain", bool(s.get("speaker_uncertain", False))),
                    "speaker_diarization_source": self._fused_final_status.get(int(s["id"]), {}).get(
                        "speaker_diarization_source", (
                            "nemotron3" if self._independent_published.get(int(s["id"]), 0) > 0
                            else "nemotron3_uncertain" if int(s["id"]) in self._independent_published
                            else s.get("speaker_diarization_source", "camplus")
                        )),
                    "speaker_final_resolution_status": self._fused_final_status.get(int(s["id"]), {}).get(
                        "speaker_final_resolution_status", s.get("speaker_final_resolution_status")),
                    "timestamp_quality": s.get("timestamp_quality", "segment"),
                    "words": s.get("words", []),
                    "translation": s.get("translation", ""),
                    "revision": s.get("revision", 0),
                    "manually_corrected": bool(s.get("manually_corrected", False)),
                    "speaker_override": bool(s.get("speaker_override", False)),
                    "speaker_override_id": s.get("speaker_override_id"),
                    "speaker_override_name": s.get("speaker_override_name"),
                    "speaker_profile_id": s.get("speaker_profile_id"),
                    "original_asr_text": s.get("original_asr_text"),
                    "corrections": s.get("corrections", []),
                }
                for s in self.segments
            ],
            "next_segment_id": self._next_segment_id,
            "speaker_map": dict(self._speaker_map),
            "ui_settings": dict(self.ui_settings),
        }

    def load_state(self, state: dict) -> None:
        """Audio transcription and speaker data handling."""
        self.segments = []
        self._speech_coverage_gaps = [
            dict(gap) for gap in state.get("speech_coverage_gaps", [])
            if isinstance(gap, dict)
        ][:256]
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
                    "saved_nemotron_id": (
                        int(item["speaker_id"])
                        if item.get("speaker_diarization_source") in (
                            "nemotron3", "nemotron3_channel_provisional",
                        )
                        and type(item.get("speaker_id")) is int
                        and item["speaker_id"] > 0
                        else None
                    ),
                    "speaker_diarization_source": item.get(
                        "speaker_diarization_source", "camplus",
                    ),
                    "speaker_acoustic_channel": item.get("speaker_acoustic_channel"),
                    "speaker_channel_gate_reason": item.get("speaker_channel_gate_reason"),
                    "speaker_activity_score": item.get("speaker_activity_score"),
                    "speaker_activity_margin": item.get("speaker_activity_margin"),
                    "speaker_activity_status": item.get("speaker_activity_status"),
                    "speaker_activity_intervals_ms": item.get(
                        "speaker_activity_intervals_ms", [],
                    ),
                    "speaker_activity_fraction": item.get("speaker_activity_fraction"),
                    "speaker_activity_time_source": item.get(
                        "speaker_activity_time_source",
                    ),
                    "caption_cues": item.get("caption_cues", []),
                    "overlap_intervals_ms": item.get("overlap_intervals_ms", []),
                    "caption_projection_status": item.get(
                        "caption_projection_status", "disabled",
                    ),
                    "speaker_uncertain": bool(item.get("speaker_uncertain", False)),
                    "text": item.get("text", "") or "",
                    "language": item.get("language", "und"),
                    "language_confidence": item.get("language_confidence"),
                    "asr_confidence": item.get("asr_confidence"),
                    "recognition_origin": item.get("recognition_origin", "primary"),
                    "speaker_match_similarity": item.get("speaker_match_similarity"),
                    "matched_profile_id": item.get("matched_profile_id"),
                    "matched_profile_name": item.get("matched_profile_name"),
                    "speaker_evidence_status": item.get("speaker_evidence_status", "unavailable"),
                    "speaker_upstream_evidence_status": item.get(
                        "speaker_upstream_evidence_status",
                    ),
                    "speaker_birth_decision": item.get("speaker_birth_decision"),
                    "speaker_final_resolution_status": item.get("speaker_final_resolution_status"),
                    "timestamp_quality": item.get("timestamp_quality", "segment"),
                    "words": item.get("words", []),
                    "translation": item.get("translation", ""),
                    "revision": int(item.get("revision", 0)),
                    "manually_corrected": bool(item.get("manually_corrected", False)),
                    "speaker_override": bool(item.get("speaker_override", False)),
                    "speaker_override_id": item.get("speaker_override_id"),
                    "speaker_override_name": item.get("speaker_override_name"),
                    "speaker_profile_id": item.get("speaker_profile_id"),
                    "original_asr_text": item.get("original_asr_text"),
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
        self._speaker_birth_approved = {
            int(s["speaker_id"]) for s in self.segments
            if type(s.get("speaker_id")) is int and s["speaker_id"] > 0
        }
        self.ui_settings = dict(state.get("ui_settings") or {})
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
        self._next_acoustic_sample = offset

        self._last_fingerprint = None
        self.meeting_title = state.get("title") or self.meeting_title

    def load_participants(self) -> None:
        """Load explicitly selected identities, or a conservative library snapshot.

        An empty participant selection enables cross-meeting matching by
        default. MEETING_AUTO_SPEAKER_MATCH=0 disables the automatic fallback.
        Auto-discovered IDs are not written into meeting participant settings.
        """
        self._auto_profile_mode = False
        self._auto_match_cache = {}
        if not self.speaker_store:
            self._participants = []
            return
        if self.participant_ids:
            self._participants = self.speaker_store.get_for_matching(self.participant_ids)
            self.participant_ids = [p["id"] for p in self._participants]
        elif (self.auto_match_saved and
              os.getenv("MEETING_AUTO_SPEAKER_MATCH", "1").strip().lower()
              in ("1", "true", "yes")):
            self._participants = self.speaker_store.get_auto_matching_profiles()
            self._auto_profile_mode = True
        else:
            self._participants = []

    def load_hotwords(self) -> None:
        """Audio transcription and speaker data handling."""
        if not self.hotword_store or not self.hotword_ids:
            self._hotword_str = ""
            self.active_hotword_count = 0
            return
        words = self.hotword_store.get_words_for_meeting(
            self.hotword_ids, limit=self._hotword_limit
        )
        # Handle transcripts and speaker assignments.
        valid_ids = {lib["id"] for lib in self.hotword_store.list_hotwords()}
        self.hotword_ids = [hid for hid in self.hotword_ids if hid in valid_ids]
        self._hotword_str = " ".join(words)
        self.active_hotword_count = len(words)
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

    def _record_untranscribed_voice(self, gap: dict) -> None:
        """Never invent missing words; retain a bounded review interval."""
        if any(
            abs(int(old["start_ms"]) - gap["start_ms"]) < 600
            and abs(int(old["end_ms"]) - gap["end_ms"]) < 1200
            for old in self._speech_coverage_gaps
        ):
            return
        self._speech_coverage_gaps.append(dict(gap))
        self._speech_coverage_gaps = self._speech_coverage_gaps[-256:]
        metrics.increment("asr_untranscribed_voice_gaps")
        self._put({
            "type": "speech_coverage",
            "gap": gap,
            "message": (
                "Voice activity had no verified transcript. "
                "Review source audio if recording is available."
            ),
        })

    # ------------------------------------------------------------------ cycle
    def _cycle(self, final: bool = False) -> bool:
        sr = self.settings.sample_rate
        start_offset, audio = self.buffer.snapshot()
        buffer_s = len(audio) / sr

        if len(audio) < sr * 0.2 and not final:
            self._send_state()
            return False
        absolute_end = start_offset + len(audio)
        if (self._speech_settings.engine == "hybrid"
                and uninformative_audio(audio, sr)):
            # Do not feed sustained microphone noise to either decoder or
            # the speaker extractor. Keep a short tail for the next genuine
            # voice onset, instead of retaining an infinite silent buffer.
            self._last_inferred_absolute_end = absolute_end
            if not final and len(audio) > 7 * sr:
                self.buffer.drop_before(len(audio) - 3 * sr)
            cleared = self._preview_tracker.clear(
                round(start_offset * 1000 / sr),
                round(absolute_end * 1000 / sr),
            )
            if cleared is not None:
                self._put({"type": "partial", **vars(cleared)})
            metrics.increment("asr_low_information_audio_suppressed")
            self._send_state()
            return False
        self._schedule_acoustic(start_offset, audio)
        # A rescan with no new microphone samples cannot improve recognition.
        # Finalization is exempt so the last incomplete utterance gets one chance.
        if not final and absolute_end == self._last_inferred_absolute_end:
            self._send_state()
            return False
        self._last_inferred_absolute_end = absolute_end

        # Handle transcripts and speaker assignments.
        phase07_asr_started = time.monotonic()
        current_content_trace = None
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
            if os.getenv("MEETING_ASR_TRACE_STAGES", "0").lower() in (
                "1", "true", "yes",
            ):
                diagnostics = recognition.recovery_diagnostics or {}
                trace = diagnostics.get("stage_trace")
                routing = diagnostics.get("hybrid")
                if ((isinstance(trace, dict) or isinstance(routing, dict))
                        and len(self._asr_stage_trace) < 1024):
                    trace = trace if isinstance(trace, dict) else {}
                    origin_ms = round(start_offset * 1000 / sr)
                    stage = {
                        "start_ms": origin_ms,
                        "end_ms": round(absolute_end * 1000 / sr),
                    }
                    for key in (
                        "vad_intervals_ms", "raw_model_intervals_ms",
                        "verified_intervals_ms", "after_recovery_intervals_ms",
                    ):
                        stage[key] = shifted(trace.get(key) or [], origin_ms)
                    stage["speech_windows"] = trace.get("speech_windows", 0)
                    stage["window_internal_vad_bypassed"] = bool(
                        trace.get("window_internal_vad_bypassed", False)
                    )
                    for name in (
                        "recovery_attempted", "recovery_confirmed",
                        "recovery_failed",
                    ):
                        stage[name] = max(0, int(trace.get(name) or 0))
                    if isinstance(routing, dict):
                        stage["hybrid_routing"] = dict(routing)
                    self._asr_stage_trace.append(stage)
                    current_content_trace = stage
        telemetry.observe_asr_cycle(buffer_s, time.monotonic() - phase07_asr_started)
        unconfirmed = []
        published_intervals_ms = []
        local_end = 0
        # Per-inference-cycle provenance for content conservation. Repeated
        # retries are counted per cycle and never called "unique words".
        commit_receipt = {
            "verified_candidates": len(sentences),
            "held": 0,
            "invalid_timestamps": 0,
            "duplicate_skipped": 0,
            "tail_echo_skipped": 0,
            "acoustic_rejected": 0,
            "language_rejected": 0,
            "published_new": 0,
            "published_from_recovery": 0,
        }
        for sentence in sentences:
            start_ms = int(sentence["start"])
            end_ms = int(sentence["end"])
            text = sentence["text"]
            language = normalize_language(sentence.get("language"))
            language_score = sentence.get("language_confidence")
            if (self._speech_settings.engine != "legacy"
                    and getattr(self, "speech_language", "auto") == "auto"
                    and (language not in PHASE1_LANGUAGE_SET | {"und", "mul"}
                         or (language == "und" and end_ms - start_ms < 1500
                             and isinstance(language_score, (int, float))
                             and language_score < .50))):
                commit_receipt["language_rejected"] += 1
                metrics.increment("asr_unverified_language_candidate")
                continue
            # Recheck the exact proposed segment's PCM. A VAD false
            # positive on a quiet, stationary interval must not be promoted
            # into a permanent caption or a new speaker cluster.
            a_local = max(0, min(len(audio), int(start_ms * sr / 1000)))
            b_local = max(0, min(len(audio), int(end_ms * sr / 1000)))
            if (self._speech_settings.engine in ("whisper", "hybrid")
                    and b_local > a_local
                    and uninformative_audio(audio[a_local:b_local], sr)):
                commit_receipt["acoustic_rejected"] += 1
                local_end = max(local_end, b_local)
                metrics.increment("asr_unvoiced_candidate_rejected")
                continue
            # Reconcile the new ASR hypothesis with already confirmed speech
            # BEFORE displaying a provisional caption. Otherwise a trailing
            # "Man." can appear as live text until the seven-second commit gate.
            s_g = start_offset + int(start_ms * sr / 1000)
            e_g = start_offset + int(end_ms * sr / 1000)
            if e_g <= s_g or start_ms < 0 or end_ms > buffer_s * 1000 + 250:
                commit_receipt["invalid_timestamps"] += 1
                continue
            if is_temporal_duplicate(s_g, e_g, text, self.segments):
                commit_receipt["duplicate_skipped"] += 1
                metrics.increment("asr_temporal_duplicate_skipped")
                local_end = max(local_end, e_g - start_offset)
                continue
            if (self._speech_settings.engine in ("whisper", "hybrid")
                    and is_adjacent_tail_echo(
                        s_g, e_g, text, self.segments, sample_rate=sr,
                        timestamp_quality=sentence.get(
                            "timestamp_quality", "segment"
                        ),
                    )):
                commit_receipt["tail_echo_skipped"] += 1
                metrics.increment("asr_adjacent_tail_echo_skipped")
                local_end = max(local_end, e_g - start_offset)
                continue

            should_hold = not final and (
                buffer_s - end_ms / 1000.0 < self._trusted_threshold_s
                or (
                    self._speech_settings.engine != "legacy"
                    and buffer_s < self._min_commit_context_s
                )
                or (
                    self._speech_settings.engine in ("whisper", "hybrid")
                    and unclosed_voice_turn(
                        start_ms, end_ms, len(audio) * 1000 // sr,
                        recognition.speech_intervals_ms,
                    )
                )
            )
            if should_hold:
                unconfirmed.append(sentence)
                commit_receipt["held"] += 1
                continue

            seg_audio = audio[s_g - start_offset : e_g - start_offset]
            speaker_task = None
            emb = None
            if self._speech_settings.engine == "legacy":
                if len(seg_audio) >= int(sr * 1.2):
                    try:
                        emb = normalized_embedding(self.models.embed(seg_audio))
                    except Exception:
                        logger.exception("Speaker embedding failed; preserving ASR text")
            elif len(seg_audio) >= int(sr * self._speaker_min_audio_s):
                speaker_task = submit_speaker_analysis(
                    sentence, seg_audio, self.models.embed, sr,
                    minimum_embedding_seconds=self._speaker_min_audio_s,
                )
            deferred = False
            if (self._speech_settings.engine != "legacy"
                    and speaker_task is None
                    and len(seg_audio) >= int(sr * self._speaker_min_audio_s)):
                deferred = self._defer_speaker_analysis(
                    self._next_segment_id, sentence, seg_audio,
                )
                if not deferred:
                    self._warn_speaker_backpressure()
            record = {
                "id": self._next_segment_id,
                "_audio_base_sample": start_offset,
                "start_sample": s_g,
                "end_sample": e_g,
                "text": text,
                "language": sentence.get("language", "und"),
                "language_confidence": sentence.get("language_confidence"),
                "asr_confidence": sentence.get("confidence"),
                "recognition_origin": sentence.get("recognition_origin", "primary"),
                "timestamp_quality": sentence.get("timestamp_quality", "segment"),
                "words": sentence.get("words", []),
                "emb": emb,
                "cluster": -1,
                "speaker_uncertain": (
                    speaker_task is not None or deferred
                    or (self._speech_settings.engine != "legacy" and emb is None)
                ),
                "speaker_evidence_status": (
                    "direct_embedding" if emb is not None
                    else "pending_turn_analysis" if speaker_task is not None
                    else "deferred_turn_analysis" if deferred
                    else "short_turn" if len(seg_audio) < int(sr * self._speaker_min_audio_s)
                    else "speaker_queue_exhausted"
                ),
            }
            self.segments.append(record)
            published_intervals_ms.append((start_ms, end_ms))
            commit_receipt["published_new"] += 1
            if record["recognition_origin"] == "gap_recovery":
                commit_receipt["published_from_recovery"] += 1
            self._next_segment_id += 1
            if (self._ecapa_shadow is not None and len(seg_audio) >= sr * 2):
                # Keep a small, temporary audio sample until CAM++ assigns
                # a stable published label. No transcript or identity is sent.
                if len(self._ecapa_shadow_audio) >= 16:
                    self._ecapa_shadow_audio.pop(
                        next(iter(self._ecapa_shadow_audio))
                    )
                # Retain a bounded excerpt in RAM until CAM++ assigns
                # a label. The shadow selector can then find voiced audio
                # later in the utterance instead of taking the first 6s.
                self._ecapa_shadow_audio[record["id"]] = np.array(
                    seg_audio[:sr * 20], dtype=np.float32, copy=True,
                )
            if speaker_task is not None:
                self._speaker_tasks[record["id"]] = speaker_task
                speaker_task.add_done_callback(self._notify_async_ready)
            else:
                self._enqueue_translation(record)
            logger.info("[%s] confirmed segment %d (%.2f ~ %.2f seconds)",
                        self.session_id[:8], record["id"], s_g / sr, e_g / sr)
            metrics.increment("asr_segments_confirmed")
            local_end = max(local_end, e_g - start_offset)

        if current_content_trace is not None:
            if sum(commit_receipt[key] for key in (
                "held", "invalid_timestamps", "duplicate_skipped",
                "tail_echo_skipped", "acoustic_rejected", "language_rejected", "published_new",
            )) != commit_receipt["verified_candidates"]:
                raise AssertionError("ASR candidate accounting not conserved")
            current_content_trace["commit_receipt"] = commit_receipt

        if self._speech_settings.engine != "legacy" and not self._speech_settings.native_streaming:
            if final:
                # An unfinished draft must not survive after final ASR.
                cleared = self._preview_tracker.clear(
                    start_offset * 1000 // sr, absolute_end * 1000 // sr)
                if cleared is not None:
                    self._put({"type": "partial", **vars(cleared)})
            else:
                preview_text = " ".join(str(s.get("text", "")) for s in unconfirmed).strip()
                if preview_text:
                    first = min(max(0, int(s["start"])) for s in unconfirmed)
                    last = max(int(s["end"]) for s in unconfirmed)
                    latest = unconfirmed[-1]
                    event = self._preview_tracker.update(
                        preview_text, latest.get("language", "und"),
                        start_offset * 1000 // sr + first,
                        start_offset * 1000 // sr + last,
                    )
                    if event is not None:
                        self._put({"type": "partial", **vars(event)})
                else:
                    # A new ASR cycle with no held words invalidates the
                    # previous draft even when VAD stays positive. Otherwise
                    # "You." can remain on screen after the decoder retracts it.
                    # Speaker-only refreshes never enter this branch.
                    cleared = self._preview_tracker.clear(
                        start_offset * 1000 // sr, absolute_end * 1000 // sr)
                    if cleared is not None:
                        self._put({"type": "partial", **vars(cleared)})
        if final and (self._speech_settings.enable_events or self._speech_settings.enable_overlap):
            try:
                from speech.analysis import analyze
                result = analyze(audio, self._speech_settings)
                base_ms = round(start_offset * 1000 / sr)
                result["audio_window_start_ms"] = base_ms
                result["audio_window_end_ms"] = round(absolute_end * 1000 / sr)
                result["covers_entire_meeting"] = start_offset == 0
                # The final buffer is normally just the remaining tail.
                # Convert segment-local overlap candidates to meeting time;
                # never imply they are an all-meeting DER prediction.
                for key in ("overlap", "separated_candidates"):
                    for event in result.get(key, []):
                        event["start_ms"] += base_ms
                        event["end_ms"] += base_ms
                self._put({"type": "audio_analysis", "result": result})
            except Exception as exc:
                self._put({"type": "feature_warning", "feature": "audio_analysis", "message": str(exc)})
        # Preserve VAD-positive intervals that have no transcript for a
        # bounded number of complete inference opportunities. An ASR failure
        # cannot silently authorize dropping earlier speech just because
        # later speech was decoded successfully.
        coverage_limited = False
        if (self._speech_settings.engine != "legacy"
                and (quality_candidate_enabled()
                     or os.getenv("MEETING_ASR_COVERAGE_RETENTION", "1")
                     .strip().lower() in ("1", "true", "yes"))):
            voice_ranges = tuple(getattr(
                recognition, "speech_intervals_ms", (),
            ))
            if voice_ranges:
                observed = []
                # Existing confirmed words may still fall inside the retained
                # rolling window and must count as covered evidence.
                for previous in self.segments:
                    a = (int(previous["start_sample"]) - start_offset) * 1000 // sr
                    b = (int(previous["end_sample"]) - start_offset) * 1000 // sr
                    if b > 0 and a < len(audio) * 1000 // sr:
                        observed.append(Utterance(
                            str(previous.get("text") or ""),
                            max(0, a), min(len(audio) * 1000 // sr, b),
                        ))
                if len(self._coverage_hold_attempts) > 2048:
                    # Absolute keys older than the current retained audio
                    # cannot affect future reclamation decisions.
                    floor = (start_offset * 1000 // sr) // 250
                    self._coverage_hold_attempts = {
                        key: value for key, value
                        in self._coverage_hold_attempts.items()
                        if key >= floor
                    }
                proposed_drop = local_end
                local_end, exhausted = choose_safe_drop(
                    proposed_drop_samples=local_end,
                    audio_samples=len(audio),
                    sample_rate=sr,
                    global_start_samples=start_offset,
                    speech_intervals_ms=voice_ranges,
                    utterances=tuple(observed),
                    attempts=self._coverage_hold_attempts,
                    final=final,
                    min_uncovered_ms=350,
                )
                coverage_limited = local_end < proposed_drop
                for gap in exhausted:
                    self._record_untranscribed_voice(gap)
        # VAD provenance from Whisper and Hybrid can safely close a
        # completed speech island. A committed text timestamp alone cannot
        # prove voice-off; without VAD intervals the buffer cursor is unchanged.
        # After the VAD speech-off event settles, retire the complete
        # acoustic island only when the same cycle committed enough aligned
        # content, no pending words lie in it, and no other voice is nearby.
        # The retention guard takes precedence over this optimization.
        if (self._speech_settings.engine in ("whisper", "hybrid")
                and local_end > 0
                and commit_receipt["published_new"] > 0
                and not final and not coverage_limited
                and recognition.speech_intervals_ms):
            verified_end = completed_island_drop(
                local_end, len(audio), recognition.speech_intervals_ms,
                published_intervals_ms=published_intervals_ms,
                held_intervals_ms=(
                    (int(s["start"]), int(s["end"])) for s in unconfirmed
                ),
                sample_rate=sr,
            )
            if verified_end > local_end:
                metrics.increment("asr_completed_voice_islands_consumed")
                local_end = verified_end
        # Handle transcripts and speaker assignments.
        if current_content_trace is not None:
            current_content_trace["candidate_intervals_ms"] = [
                [start_offset * 1000 // sr + int(item["start"]),
                 start_offset * 1000 // sr + int(item["end"])]
                for item in sentences
            ]
            current_content_trace["published_intervals_ms"] = [
                [start_offset * 1000 // sr + a,
                 start_offset * 1000 // sr + b]
                for a, b in published_intervals_ms
            ]
            current_content_trace["pending_intervals_ms"] = [
                [start_offset * 1000 // sr + int(item["start"]),
                 start_offset * 1000 // sr + int(item["end"])]
                for item in unconfirmed
            ]
        if local_end > 0:
            # Preserve any earlier unfinished utterance before advancing audio.
            if unconfirmed:
                first_held_ms = min(max(0, int(item["start"])) for item in unconfirmed)
                local_end = min(local_end, first_held_ms * sr // 1000)
            if local_end > 0:
                self.buffer.drop_before(local_end)
        elif (not final and not unconfirmed
              and self._speech_settings.engine != "legacy"
              and recognition.speech_detected is True
              and os.getenv("MEETING_ASR_SPEECH_BACKPRESSURE_GUARD", "1")
                  .strip().lower() in ("1", "true", "yes")):
            # If a model repeatedly hears speech but decodes no words,
            # never let the rolling buffer fill until WebSocket shutdown.
            # Keep the newest context; older undecodable voice is explicitly
            # flagged for source-WAV review before it leaves RAM.
            max_pending_s = min(
                45.0, max(8.0, self.buffer.capacity_samples() / sr * 0.60)
            )
            if buffer_s >= max_pending_s:
                keep_s = min(12.0, max(4.0, max_pending_s * 0.30))
                reclaim = max(0, len(audio) - int(keep_s * sr))
                if reclaim:
                    for a, b in tuple(getattr(
                        recognition, "speech_intervals_ms", (),
                    )):
                        clipped_end = min(b, reclaim * 1000 // sr)
                        if clipped_end - a >= 1000:
                            self._record_untranscribed_voice({
                                "start_ms": start_offset * 1000 // sr + a,
                                "end_ms": start_offset * 1000 // sr + clipped_end,
                                "status": "voice_detected_transcript_unavailable",
                            })
                    self.buffer.drop_before(reclaim)
                    metrics.increment("asr_untranscribed_audio_reclaimed")
        elif not final and not sentences and self._speech_settings.engine != "legacy" and recognition.speech_detected is False:
            # Reclaim only very-low-energy LEADING silence when ASR returned
            # no utterances. Preserve all uncertain or voice-like samples.
            if os.getenv("MEETING_SAFE_SILENCE_RECLAIM", "1").strip().lower() in ("1", "true", "yes"):
                from speech.silence_retention import safe_silence_prefix
                drop = safe_silence_prefix(audio, sr)
                if drop:
                    self.buffer.drop_before(drop)
                    metrics.increment("audio_safe_silence_reclaimed")

        if current_content_trace is not None:
            current_content_trace["buffer_release_until_ms"] = (
                self.buffer.start_offset() * 1000 // sr
            )
        changed = self._refresh_auxiliary_results()
        if not final:
            self._send_state()
        return changed

    def _refresh_auxiliary_results(self) -> bool:
        speaker_updated = self._collect_speakers()
        acoustic_updated = self._collect_acoustic()
        self._collect_translations()
        if not self.segments:
            return False
        try:
            if (speaker_updated or acoustic_updated
                    or self._last_reclustered_segment_id != self._next_segment_id - 1):
                self._recluster()
                self._last_reclustered_segment_id = self._next_segment_id - 1
        except Exception:
            logger.exception("Speaker clustering failed; publishing confirmed text")
            self._fallback_labels()
        changed = self._rebuild_and_push()
        self._autosave_if_due(changed)
        return changed

    def _schedule_acoustic(self, start_offset: int, audio: np.ndarray) -> None:
        """Schedule speaker evidence from continuous audio, without ASR boundaries."""
        sr = self.settings.sample_rate
        embed = getattr(self.models, "embed", None)
        if not callable(embed):
            return
        end = start_offset + len(audio)
        self._next_acoustic_sample = max(self._next_acoustic_sample, start_offset)
        # Never allocate an unbounded acoustic task backlog.
        while (self._next_acoustic_sample + self._acoustic_window_samples <= end
               and len(self._acoustic_tasks) < 10):
            start = self._next_acoustic_sample
            left = start - start_offset
            right = left + self._acoustic_window_samples
            audio_window = audio[left:right]
            task = submit_acoustic_window(start, audio_window, embed, sr)
            if task is None:
                break
            self._acoustic_tasks[start] = task
            task.add_done_callback(self._notify_async_ready)
            self._next_acoustic_sample += self._acoustic_stride_samples

    def _collect_acoustic(self, wait_seconds: float = 0.0) -> bool:
        """Backfill short ASR segments from the independent acoustic timeline."""
        if wait_seconds and self._acoustic_tasks:
            wait(tuple(self._acoustic_tasks.values()), timeout=wait_seconds)
        added = False
        for key, task in tuple(self._acoustic_tasks.items()):
            if not task.done():
                continue
            self._acoustic_tasks.pop(key, None)
            try:
                start, length, emb = task.result()
            except Exception:
                logger.exception("Independent acoustic speaker window failed")
                continue
            if self._acoustic_timeline.add(start, start + length, emb):
                added = True
        # A segment's asynchronous turn analysis can finish AFTER the
        # acoustic windows that cover it. Reconcile against existing windows
        # even when no new acoustic result arrived on this processor cycle.
        # Adding a window alone does not change any published speaker
        # embedding. Trigger reclustering only if backfill actually updates
        # a segment; otherwise new 1.5s windows cause needless churn.
        return self._backfill_acoustic_missing()

    def _backfill_acoustic_missing(self) -> bool:
        """Use existing acoustics for unmatched segments after async work.

        Keep the timeline's coverage/consistency checks unchanged. Never
        replace a valid embedding, a manual correction or an in-flight turn.
        """
        updated = False
        for segment in self.segments:
            if segment.get("speaker_override"):
                continue
            short_probe = (
                segment.get("speaker_evidence_status") == "short_probe"
                and segment.get("speaker_uncertain")
            )
            if segment.get("emb") is not None and not short_probe:
                continue
            if segment["id"] in self._speaker_tasks:
                continue
            embedding = self._acoustic_timeline.embedding_for(
                int(segment["start_sample"]), int(segment["end_sample"]),
            )
            if embedding is None:
                continue
            if short_probe:
                candidate = normalized_embedding(segment.get("emb"))
                if candidate is None or candidate.shape != embedding.shape:
                    continue
                norm = float(np.linalg.norm(candidate))
                if norm <= 1e-8 or not np.isfinite(norm):
                    continue
                similarity = float(np.dot(candidate / norm, embedding))
                # Do not promote an uncertain <1.2s candidate on its
                # own; require independent, time-aligned CAM++ support.
                if not np.isfinite(similarity) or similarity < 0.82:
                    continue
                segment["emb"] = normalized_embedding(
                    candidate / norm + embedding
                )
                segment["speaker_evidence_status"] = "short_probe_corroborated"
            else:
                segment["emb"] = embedding
                segment["speaker_evidence_status"] = "acoustic_backfill"
            segment["speaker_uncertain"] = False
            updated = True
            metrics.increment("speaker_acoustic_timeline_resolved")
        if updated:
            self._last_reclustered_segment_id = 0
        return updated

    def _notify_async_ready(self, _future: Future) -> None:
        self._async_ready.set()
        self.buffer.wake()

    def _enqueue_translation(self, segment: dict) -> None:
        if not self.translate_target:
            return
        source = str(segment.get("language") or "und").lower()
        if source == self.translate_target:
            segment["translation"] = segment["text"]
            return
        task = submit_translation(segment["text"], source, self.translate_target)
        if task is None:
            self._put({"type": "feature_warning", "feature": "translation",
                       "message": "Translation backlog is full; original text retained"})
            return
        self._translation_tasks[segment["id"]] = task
        task.add_done_callback(self._notify_async_ready)

    def _warn_speaker_backpressure(self) -> None:
        """Warn once about evidence actually dropped, not a retriable delay."""
        if self._speaker_backpressure_warned:
            return
        self._speaker_backpressure_warned = True
        metrics.increment("speaker_queue_evidence_unavailable")
        self._put({
            "type": "feature_warning",
            "feature": "diarization",
            "message": (
                "Speaker evidence backlog exceeded its bounded retry budget. "
                "Affected captions remain unidentified; ASR text is preserved."
            ),
        })

    def _defer_speaker_analysis(
        self, sid: int, sentence: dict, audio: np.ndarray,
    ) -> bool:
        """Retain bounded PCM once; retry with unchanged ASR/timestamps."""
        if (len(self._speaker_deferred) >= self._speaker_deferred_limit
                or not 0 < len(audio) <= self._speaker_deferred_max_samples):
            return False
        if sid in self._speaker_tasks or sid in self._speaker_deferred:
            return False
        self._speaker_deferred[sid] = (
            dict(sentence), np.array(audio, dtype=np.float32, copy=True),
        )
        metrics.increment("speaker_queue_deferred")
        return True

    def _retry_deferred_speakers(self) -> None:
        """Admit older evidence first without blocking real-time ASR."""
        if not self._speaker_deferred:
            return
        live_ids = {seg["id"] for seg in self.segments}
        for sid, (sentence, audio) in tuple(self._speaker_deferred.items()):
            if sid not in live_ids:
                self._speaker_deferred.pop(sid, None)
                continue
            task = submit_speaker_analysis(
                sentence, audio, self.models.embed, self.settings.sample_rate,
                minimum_embedding_seconds=self._speaker_min_audio_s,
            )
            if task is None:
                break  # Shared worker queue is still full.
            self._speaker_deferred.pop(sid, None)
            self._speaker_tasks[sid] = task
            original = next(seg for seg in self.segments if seg["id"] == sid)
            original["speaker_evidence_status"] = "pending_turn_analysis"
            task.add_done_callback(self._notify_async_ready)
            metrics.increment("speaker_queue_retried")

    def _collect_speakers(self, wait_seconds: float = 0.0) -> bool:
        if wait_seconds > 0 and self._speaker_tasks:
            wait(tuple(self._speaker_tasks.values()), timeout=wait_seconds)
        updated = False
        for sid, future in tuple(self._speaker_tasks.items()):
            if not future.done():
                continue
            self._speaker_tasks.pop(sid, None)
            original = next((s for s in self.segments if s["id"] == sid), None)
            if original is None:
                continue
            try:
                pieces = future.result()
            except Exception:
                logger.exception("Speaker-turn analysis failed; keeping transcript")
                pieces = []
            if not pieces:
                original["speaker_uncertain"] = True
                original["speaker_evidence_status"] = "turn_analysis_unavailable"
                self._enqueue_translation(original)
                updated = True
                continue
            # Async speaker analysis is advisory. Reject a malformed or
            # non-conserving turn split before mutating *any* confirmed ASR
            # segment. Otherwise a late worker can delete real words or
            # fabricate new intervals after the source audio was dropped.
            sr = self.settings.sample_rate
            base = int(original.get("_audio_base_sample", 0))
            source_start_ms = round(
                (int(original["start_sample"]) - base) * 1000 / sr
            )
            source_end_ms = round(
                (int(original["end_sample"]) - base) * 1000 / sr
            )
            if not valid_turn_partition(
                source_start_ms, source_end_ms, str(original.get("text", "")),
                pieces,
            ):
                logger.warning(
                    "Rejecting non-conserving speaker split for segment %s", sid
                )
                original["speaker_uncertain"] = True
                original["speaker_evidence_status"] = "turn_split_rejected"
                self._enqueue_translation(original)
                updated = True
                metrics.increment("speaker_turn_split_rejected")
                continue
            # Content-first mode has already published the full ASR caption.
            # Splitting the original record now would rewrite its text (and
            # timestamps) under the same segment ID, violating the confirmed
            # caption stability contract. Keep acoustic turn evidence separate;
            # the independent Nemotron caption-cue projection can annotate
            # speaker changes without replacing canonical speech content.
            if (len(pieces) > 1 and os.getenv(
                    "MEETING_CONTENT_FIRST_FUSION", "1"
            ).strip().lower() in ("1", "true", "yes")):
                self._ecapa_shadow_audio.pop(sid, None)
                original["speaker_turn_candidate_intervals_ms"] = [
                    {"start_ms": int(piece["start"]),
                     "end_ms": int(piece["end"])}
                    for piece in pieces
                ]
                compatible_voice = consistent_partition_embedding(pieces)
                if compatible_voice is not None:
                    # Acoustic cuts can arise from pauses or model flicker,
                    # even when the full ASR turn contains one human voice.
                    # Retain a shared embedding only after all parts agree.
                    original["emb"] = compatible_voice
                    original["cluster"] = -2
                    original["speaker_uncertain"] = False
                    original["speaker_evidence_status"] = (
                        "multi_turn_same_voice_supported"
                    )
                else:
                    original["speaker_uncertain"] = True
                    original["speaker_evidence_status"] = (
                        "multi_turn_acoustic_evidence_only"
                    )
                    original["emb"] = None
                    original["cluster"] = -3
                self._enqueue_translation(original)
                updated = True
                metrics.increment("speaker_async_content_first_retained")
                continue
            if len(pieces) > 1:
                # The original utterance may contain multiple speakers.
                # Never feed its unsplit audio to the ECAPA shadow worker.
                self._ecapa_shadow_audio.pop(sid, None)
                additional_turns = len(pieces) - 1
                self._acoustic_turn_splits += additional_turns
                metrics.increment("speaker_acoustic_turn_splits", additional_turns)
            source_words = original.get("words", [])
            source_language = original.get("language", "und")
            for index, piece in enumerate(pieces):
                if index == 0:
                    segment = original
                else:
                    segment = dict(original)
                    segment["id"] = self._next_segment_id
                    self._next_segment_id += 1
                    self.segments.append(segment)
                segment["start_sample"] = int(piece["start"] * self.settings.sample_rate / 1000) + (
                    original.get("_audio_base_sample", 0)
                )
                segment["end_sample"] = int(piece["end"] * self.settings.sample_rate / 1000) + (
                    original.get("_audio_base_sample", 0)
                )
                segment["text"] = piece["text"]
                segment["language"] = source_language
                segment["emb"] = normalized_embedding(piece.get("emb"))
                segment["cluster"] = -1
                segment["speaker_uncertain"] = bool(
                    piece.get("ambiguous") or piece.get("short_probe")
                    or segment["emb"] is None
                )
                segment["speaker_evidence_status"] = (
                    "short_probe" if piece.get("short_probe")
                    else "turn_voiceprint" if segment["emb"] is not None
                    else "turn_ambiguous" if piece.get("ambiguous")
                    else "turn_no_voiceprint"
                )
                segment["translation"] = ""
                if isinstance(source_words, list):
                    segment["words"] = [
                        w for w in source_words if isinstance(w, dict)
                        and piece["start"] <= (int(w.get("start_ms", 0)) + int(w.get("end_ms", 0))) / 2 < piece["end"]
                    ]
                self._enqueue_translation(segment)
            updated = True
            metrics.increment("speaker_async_completed")
        if updated:
            self._last_reclustered_segment_id = 0
        self._retry_deferred_speakers()
        return updated

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
    def _recluster(self, *, allow_historical_relabel: bool = False) -> None:
        mode = resolve_diarization_mode(
            os.getenv("MEETING_DIARIZATION_MODE")
        )
        # Explicit single-speaker mode is intended for one-person test meetings.
        if mode == "single":
            for record in self.segments:
                record["cluster"] = 1
                record["speaker_id"] = 1
                record["speaker_name"] = "Speaker 1"
            return
        if mode == "centroid":
            from speech.diarization import SpeakerCentroidTracker
            tracker = SpeakerCentroidTracker(threshold=float(os.getenv("MEETING_SPEAKER_SIMILARITY", "0.72")))
            for record in self.segments:
                stable = record.get("speaker_id")
                if isinstance(stable, int) and stable > 0 and record.get("emb") is not None:
                    tracker.seed(stable, record["emb"])
            for record in self.segments:
                record["matched_profile_id"] = None
                record["matched_profile_name"] = None
                record["speaker_match_similarity"] = None
                emb = record.get("emb")
                if emb is None:
                    record.setdefault("speaker_id", -1)
                    record.setdefault("speaker_name", "Unidentified Speaker")
                    continue
                pid, similarity = self._match_participant(emb)
                if pid is not None and similarity >= (
                    max(self.spk_threshold, float(os.getenv(
                        "MEETING_AUTO_SPEAKER_SIMILARITY", "0.78",
                    ))) if self._auto_profile_mode else self.spk_threshold
                ):
                    record["matched_profile_id"] = pid
                    record["matched_profile_name"] = next(
                        (p["name"] for p in self._participants if p["id"] == pid), "Participant"
                    )
                    record["speaker_match_similarity"] = float(similarity)
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
        mature_labels = {} if allow_historical_relabel else collect_mature_labels(
            self.segments, self.settings.sample_rate,
            self._speaker_relabel_horizon_s,
        )
        matched: dict[int, str] = {}
        pool: list[int] = []
        available_names = {p["id"]: p["name"] for p in getattr(self, "_participants", [])}
        for i, seg in enumerate(self.segments):
            seg["speaker_match_similarity"] = None
            seg["matched_profile_id"] = None
            seg["matched_profile_name"] = None
            if (seg["emb"] is None
                    or (seg.get("speaker_evidence_status") in ("short_probe", "short_probe_known_voice")
                        and seg.get("speaker_uncertain"))):
                # A lone short probe must not create or hijack an identity.
                seg["cluster"] = UNIDENTIFIED_CLUSTER
                continue
            if self._auto_profile_mode and seg["id"] in self._auto_match_cache:
                participant_id, sim = self._auto_match_cache[seg["id"]]
            else:
                participant_id, sim = self._match_participant(seg["emb"])
                if self._auto_profile_mode:
                    self._auto_match_cache[seg["id"]] = (participant_id, sim)
            if participant_id is not None and sim >= (
                max(self.spk_threshold, float(os.getenv(
                    "MEETING_AUTO_SPEAKER_SIMILARITY", "0.78",
                ))) if self._auto_profile_mode else self.spk_threshold
            ):
                matched[seg["id"]] = participant_id
                seg["speaker_match_similarity"] = float(sim)
                seg["matched_profile_id"] = participant_id
                seg["matched_profile_name"] = available_names[participant_id]
                seg["cluster"] = -1
            else:
                seg["cluster"] = -2
                pool.append(i)

        # The standard global acoustic clustering remains the default.
        if pool:
            embs = np.stack([self.segments[i]["emb"] for i in pool]).astype(np.float32)
            detailed_cluster = getattr(
                self.models, "cluster_with_info", None
            )
            if callable(detailed_cluster):
                labels, receipt = detailed_cluster(embs)
                self._last_clustering_receipt = dict(receipt)
            else:
                labels = self.models.cluster(embs)
                self._last_clustering_receipt = {
                    "strategy_requested": "external_model_adapter",
                    "strategy_executed": "model_cluster",
                    "input_embeddings": len(embs),
                    "cluster_count": len(set(map(int, labels))),
                    "fallback_reason": None,
                }
            for idx, label in zip(pool, labels):
                self.segments[idx]["cluster"] = int(label)

        # Speaker birth is an identity decision, not a raw CAM++ cluster
        # numbering decision. Apply the same voice-evidence gate to Whisper,
        # Hybrid, and Qwen before mapping raw clusters to visible IDs.
        # This replaces the hybrid-only tentative gate and the post-hoc
        # rapid-fragmentation heuristic, which allowed transient Speaker 2+
        # labels to be published and then sometimes erased.
        births = gate_new_cluster_births(
            self.segments, sample_rate=self.settings.sample_rate,
        )
        metrics.increment("speaker_birth_clusters_reused", births["reused"])
        metrics.increment("speaker_birth_segments_deferred", births["deferred"])
        metrics.increment("speaker_birth_clusters_confirmed", births["confirmed"])
        self._remap_speakers()
        self._assign_labels(matched)
        # Do not rewrite a user's already displayed historical transcript
        # on every later global clustering pass. Recent and uncertain
        # segments still receive new evidence and revisions.
        changed_historical = restore_mature_labels(
            self.segments, mature_labels)
        if changed_historical:
            metrics.increment("speaker_historical_relabels_prevented",
                              changed_historical)
        # Short probes cannot establish new people, but can safely reuse an
        # earlier strong voiceprint. Their ASR words and source time remain
        # unchanged and the identity remains explicitly provisional.
        recovered = recover_short_turn_identity(
            self.segments, sample_rate=self.settings.sample_rate,
        )
        if recovered:
            metrics.increment("speaker_short_voice_recovered", recovered)

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
        if getattr(self, "_auto_profile_mode", False):
            return match_saved_voice(
                emb, self._participants,
                minimum_similarity=float(os.getenv(
                    "MEETING_AUTO_SPEAKER_SIMILARITY", "0.78",
                )),
                minimum_margin=float(os.getenv(
                    "MEETING_AUTO_SPEAKER_MARGIN", "0.08",
                )),
            )
        best_pid: Optional[str] = None
        best_sim = -1.0
        for p in self._participants:
            for vec in p["vectors"]:
                # Do not mix ECAPA vectors with CAM++ turn embeddings.
                if vec.get("embedding_model", "camplus") != "camplus":
                    continue
                reference = np.asarray(vec["emb"], dtype=np.float32).reshape(-1)
                if reference.shape != np.asarray(emb).shape:
                    continue
                sim = float(np.dot(emb, reference))
                if not np.isfinite(sim):
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_pid = p["id"]
        # Opt-in ambiguity guard across distinct enrolled participants.
        margin = float(os.getenv("MEETING_PHASE07_SPEAKER_MARGIN", "0"))
        if margin > 0 and self._participants:
            return find_distinct_participant(self._participants, emb, margin)
        return best_pid, best_sim

    def _next_free_speaker_id(self, used: Optional[set[int]] = None) -> int:
        if used is None:
            used = (
                set(self._participant_label_map.values())
                | set(self._speaker_map.values())
                | {
                    int(s["speaker_id"]) for s in self.segments
                    if type(s.get("speaker_id")) is int and s["speaker_id"] > 0
                }
            )
        free_id = 1
        while free_id in used:
            free_id += 1
        return free_id

    def _remap_speakers(self) -> None:
        """Anchor raw cluster IDs to previously displayed speaker labels.

        Historical identities are votes, so the source cluster's numeric
        label can change without stealing a stable ID from a larger group.
        """
        self._speaker_map = stabilized_cluster_ids(
            self.segments, self._participant_label_map.values(),
            self._speaker_map,
        )
        self._speaker_seg_ids = {
            raw: {
                int(s["id"]) for s in self.segments
                if s["cluster"] == raw
            }
            for raw in self._speaker_map
        }

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
    def _submit_shadow_samples(self) -> None:
        """Send only stable CAM++-labeled, single-turn audio for observation.

        No ECAPA result can rewrite a segment, participant, or saved state.
        """
        shadow = self._ecapa_shadow
        if shadow is None or not self._ecapa_shadow_audio:
            return
        for segment in self.segments:
            sid = segment["id"]
            if sid not in self._ecapa_shadow_audio:
                continue
            # Rate-limit globally by meeting time, not a lifetime quota:
            # the final hour should receive samples as well as the first.
            # First observed voice for each CAM++ label is always eligible.
            label = segment.get("speaker_id")
            if (type(label) is not int or label <= 0
                    or segment.get("speaker_uncertain")
                    or segment.get("speaker_override")
                    or segment.get("emb") is None):
                continue
            raw_audio = self._ecapa_shadow_audio.pop(sid)
            if (label in self._ecapa_shadow_seen_labels
                    and int(segment["start_sample"]) - self._ecapa_shadow_last_sample
                        < self.settings.sample_rate * 10):
                note = getattr(shadow, "note_rate_limited", None)
                if callable(note):
                    note()
                continue
            try:
                from sessions.speaker_shadow import select_shadow_window
                audio = select_shadow_window(raw_audio)
                if audio is None:
                    note = getattr(shadow, "note_no_active_window", None)
                    if callable(note):
                        note()
                    continue
                if shadow.submit(label, audio):
                    self._ecapa_shadow_last_sample = max(
                        self._ecapa_shadow_last_sample,
                        int(segment["start_sample"]),
                    )
                    self._ecapa_shadow_seen_labels.add(label)
                    self._ecapa_shadow_sample_count += 1
            except Exception:
                logger.warning("Optional ECAPA shadow enqueue failed; CAM++ unchanged")

    def _display_nemotron_channel(
        self, channel: int, evidence_ms: int, *,
        segment_id: int, embedding: object = None,
    ) -> int | None:
        """Require independent voice evidence before adding a new speaker."""
        mapper = getattr(self, "_nemotron_channels", None)
        if mapper is None or mapper.offset != self._independent_id_offset:
            mapper = CorroboratedSpeakerChannels(offset=self._independent_id_offset)
            self._nemotron_channels = mapper
        reserved = {
            label for label in getattr(
                self, "_participant_label_map", {}
            ).values() if type(label) is int and label > 0
        }
        reserved.update(
            int(segment["speaker_override_id"])
            for segment in getattr(self, "segments", [])
            if segment.get("speaker_override")
            and type(segment.get("speaker_override_id")) is int
            and segment["speaker_override_id"] > 0
        )
        return mapper.observe(
            channel, segment_id, evidence_ms, embedding,
            reserved_labels=reserved,
        )

    def _rebuild_and_push(self) -> bool:
        """Audio transcription and speaker data handling."""
        for record in self.segments:
            if record.get("speaker_override"):
                record["speaker_id"] = int(record.get("speaker_override_id") or -1)
                record["speaker_name"] = str(record.get("speaker_override_name") or "Unidentified Speaker")
        self._submit_shadow_samples()
        ordered = sorted(self.segments, key=lambda s: s["start_sample"])
        acoustics = speaker_evidence(ordered)
        items = [
            {
                "id": s["id"],
                "speaker_id": s.get(
                    "speaker_id", self._speaker_map.get(s["cluster"], s["cluster"])
                ),
                "speaker_name": s.get("speaker_name")
                or f"Speaker {s.get('speaker_id', self._speaker_map.get(s['cluster'], s['cluster']))}",
                "matched_profile_id": s.get("matched_profile_id"),
                "matched_profile_name": s.get("matched_profile_name"),
                "speaker_match_similarity": s.get("speaker_match_similarity"),
                "speaker_evidence_status": s.get("speaker_evidence_status"),
                "start_ms": round(s["start_sample"] * 1000 / self.settings.sample_rate),
                "end_ms": round(s["end_sample"] * 1000 / self.settings.sample_rate),
                "text": s["text"],
                "language": s.get("language", "und"),
                "language_confidence": s.get("language_confidence"),
                "timestamp_quality": s.get("timestamp_quality", "segment"),
                "translation": s.get("translation", ""),
                "recognition_origin": s.get("recognition_origin", "primary"),
                "text_confidence": s.get("asr_confidence"),
                **acoustics[s["id"]],
                "revision": s.get("revision", 0),
            }
            for s in ordered
        ]
        # Keep the raw CAM++ embeddings and cluster votes intact. Publish a
        # separate primary-speaker projection only when the source-independent
        # 10 ms Nemotron activity has sufficient evidence.
        independent = getattr(self, "independent_diarization", None)
        self._independent_published = {}
        self._independent_scores = {}
        self._independent_support = {}
        self._fused_final_status = {}
        self._content_first_projection = {}
        origin = getattr(self, "_independent_pcm_origin_samples", 0)
        effective_embeddings = {}
        for item, source in zip(items, ordered):
            if source.get("speaker_override"):
                item["speaker_diarization_source"] = "manual_override"
                continue
            if source.get("matched_profile_id") and source.get("matched_profile_name"):
                # A supported registered voiceprint is an identity, while a
                # Nemotron integer is only an anonymous acoustic turn index.
                # Never let the anonymous label erase an identified person.
                item["speaker_name"] = source["matched_profile_name"]
                item["speaker_diarization_source"] = "registered_voiceprint"
                continue
            saved = source.get("saved_nemotron_id")
            if type(saved) is int and saved > 0:
                item["speaker_id"] = saved
                item["speaker_name"] = f"Speaker {saved}"
                item["speaker_diarization_source"] = (
                    "nemotron3_channel_provisional"
                    if source.get("speaker_diarization_source") == "nemotron3_channel_provisional"
                    else "nemotron3"
                )
                self._independent_published[int(item["id"])] = saved
            elif (source.get("speaker_diarization_source") in (
                      "camplus_bidirectional_recovery", "nemotron3_backfilled",
                      "nemotron3_channel_provisional")
                  and int(source["end_sample"]) <= origin
                  and type(source.get("speaker_id")) is int
                  and source["speaker_id"] > 0):
                item["speaker_id"] = source["speaker_id"]
                item["speaker_name"] = f"Speaker {source['speaker_id']}"
                restored_source = source["speaker_diarization_source"]
                item["speaker_status"] = (
                    "acoustic_backfilled_provisional"
                    if restored_source == "nemotron3_backfilled"
                    else "acoustic_channel_provisional"
                    if restored_source == "nemotron3_channel_provisional"
                    else "short_voice_recovered_provisional"
                )
                item["speaker_uncertain"] = True
                item["speaker_diarization_source"] = restored_source
                self._independent_published[int(item["id"])] = int(source["speaker_id"])
            elif (source.get("speaker_diarization_source") == "nemotron3_uncertain"
                  and int(source["end_sample"]) <= origin):
                # Old resumed captions retain their saved honest unknown,
                # even if a new session reclusters legacy CAM++ embeddings.
                item["speaker_id"] = -1
                item["speaker_name"] = "Unidentified Speaker"
                item["speaker_diarization_source"] = "nemotron3_uncertain"
                self._independent_published[int(item["id"])] = -1
        official_turns = None
        if independent is not None:
            activity = independent.activity_snapshot()
            completed_check = getattr(independent, "is_complete", None)
            independent_complete = (
                bool(completed_check()) if callable(completed_check) else False
            )
            turns_reader = getattr(independent, "final_turns_snapshot", None)
            official_turns = (
                turns_reader() if independent_complete and callable(turns_reader)
                else None
            )
            if (activity is not None and len(activity)) or official_turns is not None:
                for item, source in zip(items, ordered):
                    if (source.get("speaker_override")
                            or (source.get("matched_profile_id")
                                and source.get("matched_profile_name"))
                            or int(source["end_sample"]) <= origin):
                        # Anonymous Nemotron activity may refine unlabeled turns,
                        # but it cannot erase a registered voice identity or a
                        # human speaker correction during a later projection.
                        continue
                    start_ms = round(
                        (int(source["start_sample"]) - origin) * 1000
                        / self.settings.sample_rate
                    )
                    end_ms = round(
                        (int(source["end_sample"]) - origin) * 1000
                        / self.settings.sample_rate
                    )
                    if start_ms < 0:
                        continue
                    # Reuse independent acoustic windows when a short ASR
                    # segment has no usable voice embedding. The audio must
                    # cover the actual utterance and agree across windows.
                    # Explicit mixed-speaker evidence must not be collapsed
                    # into a single person's embedding.
                    voice_embedding = source.get("emb")
                    if (voice_embedding is None
                            and source.get("speaker_evidence_status")
                                != "multi_turn_acoustic_evidence_only"
                            and source["end_sample"] - source["start_sample"]
                                >= round(self.settings.sample_rate * 1.2)):
                        timeline = getattr(self, "_acoustic_timeline", None)
                        if timeline is not None:
                            voice_embedding = timeline.embedding_for(
                                int(source["start_sample"]),
                                int(source["end_sample"]),
                                min_coverage=0.75,
                                min_consistency=0.78,
                            )
                    if voice_embedding is not None:
                        effective_embeddings[int(item["id"])] = voice_embedding
                    # Safe content-first display projection is secondary to
                    # the canonical, never-rewritten ASR segment. Words from
                    # this inference window are relative to _audio_base_sample.
                    # Unproven word timing or reconstructed text never triggers
                    # speaker splitting. Repeated calls cannot add words.
                    if (official_turns is not None
                            and os.getenv(
                                "MEETING_CONTENT_FIRST_FUSION", "1"
                            ).strip().lower() in ("1", "true", "yes")):
                        try:
                            word_base = source.get("_audio_base_sample")
                            word_base_ms = (
                                round(word_base * 1000 / self.settings.sample_rate)
                                if type(word_base) is int and word_base >= 0
                                else None
                            )
                            projection = project_confirmed_caption(
                                text=str(source["text"]),
                                words=source.get("words"),
                                caption_start_ms=item["start_ms"],
                                caption_end_ms=item["end_ms"],
                                word_base_ms=word_base_ms,
                                official_turns=official_turns,
                                acoustic_origin_ms=round(
                                    origin * 1000 / self.settings.sample_rate
                                ),
                                speaker_id_offset=self._independent_id_offset,
                                speaker_id_map=getattr(
                                    self, "_nemotron_channels", CorroboratedSpeakerChannels()
                                ).labels,
                            )
                        except (TypeError, ValueError, OverflowError):
                            logger.warning(
                                "Content-first projection skipped: invalid acoustic"
                                " or word alignment evidence",
                                exc_info=True,
                            )
                        else:
                            # Safe metadata only: untouched original text,
                            # original timeline, source word list, and revision.
                            item["caption_cues"] = projection["cues"]
                            item["overlap_intervals_ms"] = (
                                projection["overlap_intervals_ms"]
                            )
                            item["caption_projection_status"] = (
                                projection["status"]
                            )
                            self._content_first_projection[int(item["id"])] = (
                                projection
                            )
                    if official_turns is None and activity is not None:
                        # Expose simultaneous acoustic activity while the
                        # meeting is still running, without assigning words
                        # to a source the mixed audio did not separate.
                        first = start_ms // 10
                        last = (end_ms + 9) // 10
                        local = activity[first:last]
                        if len(local) and np.any(np.sum(local >= 0.65, axis=1) >= 2):
                            try:
                                overlap = project_confirmed_caption(
                                    text=str(source["text"]), words=None,
                                    caption_start_ms=item["start_ms"],
                                    caption_end_ms=item["end_ms"], word_base_ms=None,
                                    official_turns=activity_turns(
                                        local, threshold=0.65, min_turn_ms=120,
                                    ),
                                    acoustic_origin_ms=round(
                                        origin * 1000 / self.settings.sample_rate
                                    ) + first * 10,
                                    speaker_id_map=self._nemotron_channels.labels,
                                )["overlap_intervals_ms"]
                            except (AttributeError, TypeError, ValueError, OverflowError):
                                logger.debug("Live overlap evidence unavailable", exc_info=True)
                            else:
                                item["overlap_intervals_ms"] = overlap
                                self._content_first_projection[int(item["id"])] = {
                                    "status": "live_overlap_only", "cues": [],
                                    "overlap_intervals_ms": overlap,
                                }
                    decision = (
                        choose_caption_speaker_from_turns(
                            official_turns, start_ms, end_ms,
                        )
                        if official_turns is not None
                        else choose_caption_speaker(
                            activity, start_ms, end_ms,
                        )
                    )
                    item["speaker_activity_score"] = decision.activity_score
                    item["speaker_activity_margin"] = decision.margin
                    item["speaker_activity_status"] = decision.status
                    item["speaker_acoustic_channel"] = (
                        int(decision.speaker_id)
                        if type(decision.speaker_id) is int else None
                    )
                    self._independent_scores[int(item["id"])] = {
                        "speaker_acoustic_channel": item["speaker_acoustic_channel"],
                        "speaker_activity_score": decision.activity_score,
                        "speaker_activity_margin": decision.margin,
                        "speaker_activity_status": decision.status,
                    }
                    item["speaker_score_kind"] = (
                        "nemotron_official_turn_coverage_not_probability"
                        if official_turns is not None
                        else "nemotron_frame_activity_not_identity_probability"
                    )
                    if decision.speaker_id is not None:
                        acoustic_support_ms = round(
                            decision.analyzed_ms
                            * max(0.0, float(decision.active_fraction or 0.0))
                        )
                        published = self._display_nemotron_channel(
                            decision.speaker_id, acoustic_support_ms,
                            segment_id=int(item["id"]),
                            embedding=voice_embedding,
                        )
                        gate_reader = getattr(
                            self._nemotron_channels, "gate_reason", None
                        )
                        item["speaker_channel_gate_reason"] = (
                            str(gate_reader(decision.speaker_id))
                            if callable(gate_reader) else "unavailable"
                        )
                        self._independent_scores[int(item["id"])][
                            "speaker_channel_gate_reason"
                        ] = item["speaker_channel_gate_reason"]
                        if published is None and acoustic_support_ms >= 500:
                            # A new raw channel is not a new person if its
                            # independent voice vector strongly matches a
                            # speaker already established in this meeting.
                            published = self._nemotron_channels.match_known(
                                voice_embedding
                            )
                            if published is not None:
                                item["speaker_status"] = "voice_reuse_confirmed"
                        if published is None:
                            if preserve_supported_camplus(source):
                                # A new raw channel is not sufficient reason
                                # to erase an independently supported CAM++
                                # speaker identity. Keep it provisional and
                                # leave the final projection in CAM++'s hands.
                                item["speaker_status"] = "acoustic_provisional"
                                item["speaker_diarization_source"] = (
                                    "camplus_independent_unresolved"
                                )
                                source["speaker_diarization_source"] = (
                                    "camplus_independent_unresolved"
                                )
                            else:
                                # No supported voice evidence from either
                                # backend. This is an honest unknown.
                                item["speaker_id"] = -1
                                item["speaker_name"] = "Unidentified Speaker"
                                item["speaker_status"] = "speaker_birth_pending"
                                item["speaker_diarization_source"] = (
                                    "nemotron3_uncertain"
                                )
                                self._independent_published[int(item["id"])] = -1
                            continue
                        item["speaker_id"] = published
                        item["speaker_name"] = f"Speaker {published}"
                        item["speaker_status"] = "acoustic_primary"
                        item["speaker_diarization_source"] = "nemotron3"
                        if official_turns is not None:
                            try:
                                support = caption_support_summary(
                                    official_turns, start_ms, end_ms,
                                    decision.speaker_id,
                                )
                                # Return absolute meeting time to consumers,
                                # even when the AOSC state began after reconnect.
                                offset_ms = round(
                                    origin * 1000 / self.settings.sample_rate
                                )
                                support["speaker_activity_intervals_ms"] = [
                                    [a + offset_ms, b + offset_ms]
                                    for a, b in support["speaker_activity_intervals_ms"]
                                ]
                            except ValueError:
                                # Optional metadata must never delete a valid
                                # caption or interrupt meeting finalization.
                                logger.warning(
                                    "Invalid official caption acoustic support",
                                    exc_info=True,
                                )
                            else:
                                item.update(support)
                                self._independent_support[int(item["id"])] = support
                        self._independent_published[int(item["id"])] = published
                    elif (decision.status == "acoustically_ambiguous"
                          or independent_complete):
                        # A missing or ambiguous Nemotron frame does not
                        # invalidate an independent matching voiceprint.
                        # Reuse only a speaker already corroborated from
                        # acoustic evidence, never a raw CAM++ cluster ID.
                        matched_known = self._nemotron_channels.match_known(
                            voice_embedding
                        )
                        if matched_known is not None:
                            item["speaker_id"] = matched_known
                            item["speaker_name"] = f"Speaker {matched_known}"
                            item["speaker_status"] = "known_voice_recovered"
                            item["speaker_diarization_source"] = "camplus_voice_reuse"
                            self._independent_published[int(item["id"])] = matched_known
                        elif preserve_supported_camplus(source):
                            item["speaker_status"] = "acoustic_provisional"
                            item["speaker_diarization_source"] = (
                                "camplus_independent_unresolved"
                            )
                            source["speaker_diarization_source"] = (
                                "camplus_independent_unresolved"
                            )
                        else:
                            item["speaker_id"] = -1
                            item["speaker_name"] = "Unidentified Speaker"
                            item["speaker_status"] = "uncertain"
                            item["speaker_diarization_source"] = "nemotron3_uncertain"
                            self._independent_published[int(item["id"])] = -1
                    else:
                        # Incomplete Nemotron timeline: retain provisional
                        # CAM++ output until acoustic frames arrive. This
                        # also preserves legacy fallbacks on model failure.
                        item["speaker_diarization_source"] = "camplus_provisional"
        # After the stream has established a new channel, revisit previously
        # unresolved segments of that SAME channel. Require matching CAM++
        # evidence, a strong activity share, and a strong speaker margin.
        channel_mapper = getattr(self, "_nemotron_channels", None)
        restored_channels = (
            backfill_confirmed_channels(
                items, ordered, channel_mapper.labels,
                channel_mapper.match_known,
                effective_embeddings=effective_embeddings,
            )
            if channel_mapper is not None else {}
        )
        for segment_id, label in restored_channels.items():
            self._independent_published[int(segment_id)] = int(label)
        if official_turns is not None and channel_mapper is not None:
            projected_channels = reconcile_final_channels(
                items, ordered, official_turns, channel_mapper.labels, offset=self._independent_id_offset,
            )
            for segment_id, label in projected_channels.items():
                self._independent_published[segment_id] = label
        # MEETING_FUSED_SHORT_RECOVERY_V1
        recovered_short = recover_fused_short_turns(
            items, ordered, sample_rate=self.settings.sample_rate,
        )
        for segment_id, label in recovered_short.items():
            self._independent_published[int(segment_id)] = int(label)
        # A final speaker ID and CAM++ evidence are separate facts.
        # Keep both, while persisting the displayed fusion decision.
        self._fused_final_status = {}
        for item, source in zip(items, ordered):
            ident = item.get("speaker_id")
            origin_kind = str(item.get("speaker_diarization_source") or "camplus")
            if source.get("speaker_override"):
                final_reason = "manual_override"
                uncertain = False
            elif source.get("matched_profile_id"):
                final_reason = "registered_identity"
                uncertain = False
            elif type(ident) is not int or ident <= 0:
                final_reason = "unresolved"
                uncertain = True
                item["speaker_status"] = "uncertain"
            elif origin_kind == "camplus_bidirectional_recovery":
                final_reason = "voice_recovered_provisional"
                uncertain = True
                item["speaker_status"] = "short_voice_recovered_provisional"
            elif origin_kind == "nemotron3_backfilled":
                final_reason = "independent_acoustic_backfilled_provisional"
                uncertain = True
                item["speaker_status"] = "acoustic_backfilled_provisional"
            elif origin_kind == "nemotron3_channel_provisional":
                final_reason = "independent_channel_provisional"
                uncertain = True
            elif origin_kind == "nemotron3":
                final_reason = "independent_acoustic_assigned"
                uncertain = False
                item["speaker_status"] = "acoustic_primary"
            elif origin_kind == "camplus_independent_unresolved":
                final_reason = "camplus_provisional"
                uncertain = True
            else:
                final_reason = "camplus_" + str(item.get("speaker_status") or "tentative")
                uncertain = bool(source.get("speaker_uncertain", False))
            item["speaker_uncertain"] = uncertain
            item["speaker_final_resolution_status"] = final_reason
            self._fused_final_status[int(item["id"])] = {
                "speaker_status": item.get("speaker_status", "unknown"),
                "speaker_uncertain": uncertain,
                "speaker_diarization_source": origin_kind,
                "speaker_final_resolution_status": final_reason,
            }
        phase07_speaker = speaker_stats(items)
        metrics.gauge("phase07_speaker_changes", phase07_speaker["speaker_switches"])
        metrics.gauge("phase07_unidentified_segments", phase07_speaker["unidentified_segments"])
        fingerprint = tuple(
            (i["id"], i["speaker_id"], i["speaker_name"], i["start_ms"], i["end_ms"], i["text"],
             i["language"], i.get("language_confidence"), i.get("translation"),
             i.get("recognition_origin"), i.get("text_confidence"),
             i.get("speaker_status"), i.get("speaker_similarity"),
             i.get("speaker_margin"), i.get("speaker_evidence_count"),
             i.get("speaker_diarization_source"), i.get("matched_profile_id"),
             i.get("speaker_match_similarity"), i.get("speaker_activity_score"),
             i.get("speaker_activity_status"),
             i.get("speaker_acoustic_channel"),
             i.get("speaker_channel_gate_reason"),
             i.get("speaker_final_resolution_status"),
             tuple(tuple(pair) for pair in i.get(
                 "speaker_activity_intervals_ms", [],
             )),
             tuple(
                 (cue["speaker_id"], cue["start_ms"],
                  cue["end_ms"], cue["text"])
                 for cue in i.get("caption_cues", [])
             ),
             tuple(
                 (row["start_ms"], row["end_ms"],
                  tuple(row["speaker_ids"]))
                 for row in i.get("overlap_intervals_ms", [])
             ))
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

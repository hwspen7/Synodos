"""Persist meeting metadata and resumable session state as pickle files."""

from __future__ import annotations

import json
import os
import pickle
import re
import threading
import uuid
from functools import reduce
from speech.text_format import join_transcript_text, split_display_sentences, ends_finished_thought
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_ms(ms: int) -> str:
    total = max(0, int(ms) // 1000)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class MeetingStore:
    """Store one meeting per directory, with a meeting.pkl snapshot."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()

    # ------------------------------------------------------------ paths
    def valid_id(self, meeting_id: str) -> bool:
        return bool(_ID_RE.match(meeting_id))

    def _dir(self, meeting_id: str) -> Path:
        if not _ID_RE.match(meeting_id):
            raise ValueError("invalid meeting id")
        return self.root / meeting_id

    def _read_payload(self, meeting_id: str) -> dict | None:
        """Read and deserialize the meeting.pkl payload."""
        d = self._dir(meeting_id)
        f = d / "meeting.pkl"
        if f.exists():
            try:
                return pickle.loads(f.read_bytes())
            except Exception:
                return None
        return None

    # ------------------------------------------------------------ crud
    def create(self, meeting_id: str | None = None, title: str = "") -> dict:
        """Create an empty meeting with metadata and return its metadata."""
        mid = meeting_id or uuid.uuid4().hex
        d = self._dir(mid)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": mid,
            "title": (title or "Untitled Meeting").strip() or "Untitled Meeting",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "sample_rate": 16000,
            "merge_gap_s": 2.0,
            "segments": 0,
            "duration_ms": 0,
            "speakers": 0,
        }
        self._write_pickle(d / "meeting.pkl", {"meta": meta, "state": None})
        return meta

    def exists(self, meeting_id: str) -> bool:
        return self._dir(meeting_id).joinpath("meeting.pkl").exists()

    def list_meetings(self) -> list[dict]:
        out: list[dict] = []
        if not self.root.exists():
            return out
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            payload = self._read_payload(d.name)
            if payload is None:
                continue
            meta = payload.get("meta")
            if isinstance(meta, dict):
                out.append(meta)
        out.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return out

    def get_meta(self, meeting_id: str) -> dict | None:
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        meta = payload.get("meta")
        return meta if isinstance(meta, dict) else None

    def get_meeting(self, meeting_id: str) -> dict | None:
        """Return meeting metadata and participant/hotword settings for the UI."""
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        meta = payload.get("meta") or {}
        state = payload.get("state") or {}
        return {
            "id": meeting_id,
            "title": meta.get("title") or "Untitled Meeting",
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "participant_ids": list(state.get("participant_ids") or []),
            "auto_match_saved": bool(state.get("auto_match_saved", True)),
            "hotword_ids": list(state.get("hotword_ids") or []),
            "spk_threshold": float(state.get("spk_threshold", 0.3)),
            "ui_settings": dict(state.get("ui_settings") or {}),
        }

    @staticmethod
    def _segment_summary(segment: dict, sample_rate: int) -> dict:
        """Safely expose transcript evidence, excluding voiceprint arrays."""
        s = segment
        start_ms = round(int(s["start_sample"]) * 1000 / sample_rate)
        end_ms = round(int(s["end_sample"]) * 1000 / sample_rate)
        return {
            "id": int(s["id"]),
            "text": s.get("text") or "",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_s": round(max(0.0, (end_ms - start_ms) / 1000.0), 3),
            "speaker_id": int(s.get("speaker_id", -1)),
            "speaker_name": str(s.get("speaker_name") or "Unidentified Speaker"),
            "language": s.get("language", "und"),
            "translation": s.get("translation", ""),
            "revision": int(s.get("revision", 0)),
            "speaker_profile_id": s.get("speaker_profile_id"),
            "matched_profile_id": s.get("matched_profile_id"),
            "matched_profile_name": s.get("matched_profile_name"),
            "speaker_match_similarity": s.get("speaker_match_similarity"),
            "manually_corrected": bool(s.get("manually_corrected", False)),
            "review_count": len(s.get("corrections") or []),
            "recognition_origin": s.get("recognition_origin", "primary"),
            "text_confidence": s.get("asr_confidence"),
            "timestamp_quality": s.get("timestamp_quality", "segment"),
            "speaker_status": s.get("speaker_status", "unknown"),
            "speaker_uncertain": bool(s.get("speaker_uncertain", False)),
            "speaker_similarity": s.get("speaker_similarity"),
            "speaker_margin": s.get("speaker_margin"),
            "speaker_evidence_count": s.get("speaker_evidence_count", 0),
            "speaker_evidence_status": s.get("speaker_evidence_status", "unavailable"),
            "speaker_upstream_evidence_status": s.get(
                "speaker_upstream_evidence_status",
            ),
            "speaker_birth_decision": s.get("speaker_birth_decision"),
            # Only publish a boolean, never raw private voiceprint vectors.
            "speaker_embedding_available": s.get("emb") is not None,
            "speaker_score_kind": (
                "nemotron_official_turn_coverage_not_probability"
                if str(s.get("speaker_activity_status", "")).startswith(
                    "official_turn_"
                )
                else "nemotron_frame_activity_not_identity_probability"
                if s.get("speaker_diarization_source") in (
                    "nemotron3", "nemotron3_uncertain",
                )
                else "raw_cosine_not_probability"
            ),
            "speaker_diarization_source": s.get(
                "speaker_diarization_source", "camplus",
            ),
            "speaker_activity_score": s.get("speaker_activity_score"),
            "speaker_activity_margin": s.get("speaker_activity_margin"),
            "speaker_activity_status": s.get("speaker_activity_status"),
            "speaker_activity_intervals_ms": s.get(
                "speaker_activity_intervals_ms", [],
            ),
            "speaker_activity_fraction": s.get("speaker_activity_fraction"),
            "speaker_activity_time_source": s.get("speaker_activity_time_source"),
            "caption_cues": s.get("caption_cues", []),
            "overlap_intervals_ms": s.get("overlap_intervals_ms", []),
            "caption_projection_status": s.get(
                "caption_projection_status", "disabled",
            ),
        }

    def list_segments(self, meeting_id: str) -> list[dict] | None:
        """Return complete safe evidence for later summaries, no voiceprints."""
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        meta = payload.get("meta") or {}
        state = payload.get("state") or {}
        sr = int(state.get("sample_rate", meta.get("sample_rate") or 16000))
        return [
            self._segment_summary(s, sr)
            for s in (state.get("segments") or [])
        ]

    def list_speech_coverage_gaps(self, meeting_id: str) -> list[dict] | None:
        """Expose unresolved voice intervals, never imaginary transcript text."""
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        state = payload.get("state") or {}
        return [
            dict(item)
            for item in (state.get("speech_coverage_gaps") or [])
            if isinstance(item, dict)
        ]

    # ------------------------------------------------------------ state
    def save_state(self, state: dict, *,
                   expected_store_revision: int | None = None) -> dict:
        """Atomically write a snapshot, rejecting stale reviewed states.

        The RLock serializes writes by this store in one process. For
        cross-process CAS, run the PostgreSQL-backed store; filesystem
        optimistic checks alone are not a distributed transaction.
        """
        from sessions.manual_review import ReviewConflict

        mid = state["id"]
        d = self._dir(mid)
        with self._write_lock:
            d.mkdir(parents=True, exist_ok=True)
            sr = int(state.get("sample_rate", 16000))
            if sr <= 0:
                raise ValueError("Invalid sample rate")
            segs = state.get("segments", []) or []
            duration_ms = int(
                max((int(seg["end_sample"]) for seg in segs), default=0)
                * 1000 / sr
            )
            speaker_map = state.get("speaker_map", {}) or {}
            stable_ids = {
                seg.get("speaker_id",
                        speaker_map.get(str(seg.get("cluster", -1)),
                                        seg.get("cluster", -1)))
                for seg in segs
            }
            old = self.get_meta(mid) or {}
            version = int(old.get("_store_version", 0))
            if (expected_store_revision is not None
                    and version != expected_store_revision):
                raise ReviewConflict(
                    f"Stale meeting snapshot {expected_store_revision}; current is {version}"
                )
            meta = {
                "id": mid,
                "title": (state.get("title") or "Untitled Meeting").strip()
                         or "Untitled Meeting",
                "created_at": old.get("created_at") or _now_iso(),
                "updated_at": _now_iso(),
                "sample_rate": sr,
                "merge_gap_s": float(state.get("merge_gap_s", 2.0)),
                "segments": len(segs),
                "duration_ms": duration_ms,
                "speakers": len(stable_ids),
                "_store_version": version + 1,
            }
            payload = {"meta": meta, "state": dict(state)}
            tmp = d / ("meeting-" + uuid.uuid4().hex + ".tmp")
            try:
                tmp.write_bytes(pickle.dumps(
                    payload, protocol=pickle.HIGHEST_PROTOCOL
                ))
                os.replace(tmp, d / "meeting.pkl")
            finally:
                tmp.unlink(missing_ok=True)
            return meta

    def load_state(self, meeting_id: str) -> dict | None:
        """Load complete session state to resume the meeting.

        Only text and embeddings are stored; resume starts after the last confirmed segment.
        """
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        state = dict(payload.get("state") or {})
        state["meta"] = payload.get("meta") or {}
        return state

    def delete(self, meeting_id: str) -> bool:
        d = self._dir(meeting_id)
        if not d.exists():
            return False
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        return True

    # ------------------------------------------------------------ export
    def export(self, meeting_id: str, fmt: str = "md") -> str:
        """Export the meeting transcript as Markdown or JSON."""
        payload = self._read_payload(meeting_id)
        if payload is None:
            raise FileNotFoundError(meeting_id)
        meta = payload.get("meta") or {}
        state = payload.get("state") or {}
        segments = state.get("segments") or []
        sr = int(state.get("sample_rate", meta.get("sample_rate") or 16000))
        merge_gap_s = float(state.get("merge_gap_s", meta.get("merge_gap_s") or 2.0))
        speaker_map = {int(k): int(v) for k, v in (state.get("speaker_map") or {}).items()}
        paragraphs = self._build_paragraphs(segments, sr, merge_gap_s, speaker_map)

        if fmt == "json":
            return json.dumps(
                {
                    "meta": meta,
                    "transcript": {
                        "paragraphs": paragraphs,
                        "segments": [
                            self._segment_summary(s, sr) for s in segments
                        ],
                        "speech_coverage_gaps": [
                            dict(item)
                            for item in (state.get("speech_coverage_gaps") or [])
                            if isinstance(item, dict)
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )

        lines = [f"# {meta.get('title', 'Untitled Meeting')}", ""]
        lines.append(f"- Created: {meta.get('created_at', '')}")
        lines.append(f"- Updated: {meta.get('updated_at', '')}")
        lines.append(f"- Duration: {_fmt_ms(meta.get('duration_ms', 0))}")
        lines.append(f"- Segments: {meta.get('segments', 0)}")
        lines.append(f"- Speakers: {meta.get('speakers', 0)}")
        lines.append("")
        if not paragraphs:
            lines.append("(No content yet)")
            return "\n".join(lines)
        lines.append("## Transcript")
        lines.append("")
        for i, p in enumerate(paragraphs, start=1):
            name = p.get("speaker_name") or f"Speaker {int(p.get('speaker_id', 0))}"
            lines.append(f"### {i}. {name}")
            lines.append(f"{_fmt_ms(p.get('start_ms', 0))} – {_fmt_ms(p.get('end_ms', 0))}")
            lines.append("")
            sentences = split_display_sentences(p.get("text") or "")
            lines.extend(sentences if sentences else ["(No content)"])
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_paragraphs(
        segments: list[dict],
        sr: int,
        merge_gap_s: float,
        speaker_map: dict[int, int],
    ) -> list[dict]:
        """Merge adjacent confirmed segments from the same speaker within merge_gap."""
        ordered = sorted(segments, key=lambda s: int(s.get("start_sample", 0)))
        gap_samples = int(min(max(merge_gap_s, 0.0), 0.85) * sr)
        groups: list[dict] = []
        for seg in ordered:
            cluster = int(seg.get("cluster", -1))
            # Prefer persisted speaker IDs for participants; fall back to cluster mappings.
            speaker_id = int(seg.get("speaker_id", speaker_map.get(cluster, cluster)))
            last = groups[-1] if groups else None
            if (
                last is not None
                and speaker_id > 0
                and speaker_id == last["speaker_id"]
                and not ends_finished_thought(str(last["segments"][-1].get("text") or ""))
                and (int(seg["start_sample"]) - int(last["end_sample"])) < gap_samples
            ):
                last["end_sample"] = int(seg["end_sample"])
                last["segments"].append(seg)
            else:
                groups.append(
                    {
                        "cluster": cluster,
                        "speaker_id": speaker_id,
                        "start_sample": int(seg["start_sample"]),
                        "end_sample": int(seg["end_sample"]),
                        "segments": [seg],
                    }
                )

        paragraphs: list[dict] = []
        for g in groups:
            first = g["segments"][0]
            paragraphs.append(
                {
                    "id": int(first["id"]),
                    "speaker_id": g["speaker_id"],
                    "speaker_name": first.get("speaker_name", ""),
                    "start_ms": round(g["start_sample"] * 1000 / sr),
                    "end_ms": round(g["end_sample"] * 1000 / sr),
                    "text": reduce(join_transcript_text, (str(s.get("text") or "") for s in g["segments"]), ""),
                    "sentences": [
                        line for seg in g["segments"]
                        for line in split_display_sentences(str(seg.get("text") or ""))
                    ],
                    "segments": [
                        {
                            "id": int(seg["id"]),
                            "start_ms": round(int(seg["start_sample"]) * 1000 / sr),
                            "end_ms": round(int(seg["end_sample"]) * 1000 / sr),
                        }
                        for seg in g["segments"]
                    ],
                }
            )
        return paragraphs

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _write_pickle(path: Path, payload: dict) -> None:
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

"""Persist meeting metadata and resumable session state as pickle files."""

from __future__ import annotations

import json
import os
import pickle
import re
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
            "hotword_ids": list(state.get("hotword_ids") or []),
            "spk_threshold": float(state.get("spk_threshold", 0.3)),
            "ui_settings": dict(state.get("ui_settings") or {}),
        }

    def list_segments(self, meeting_id: str) -> list[dict] | None:
        """Return segment summaries without embeddings for speaker enrollment."""
        payload = self._read_payload(meeting_id)
        if payload is None:
            return None
        meta = payload.get("meta") or {}
        state = payload.get("state") or {}
        sr = int(state.get("sample_rate", meta.get("sample_rate") or 16000))
        segs = state.get("segments") or []
        out: list[dict] = []
        for s in segs:
            start_ms = round(int(s["start_sample"]) * 1000 / sr)
            end_ms = round(int(s["end_sample"]) * 1000 / sr)
            out.append(
                {
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
                }
            )
        return out

    # ------------------------------------------------------------ state
    def save_state(self, state: dict) -> dict:
        """Atomically replace a meeting snapshot without storing raw audio."""
        mid = state["id"]
        d = self._dir(mid)
        d.mkdir(parents=True, exist_ok=True)

        sr = int(state.get("sample_rate", 16000))
        segs = state.get("segments", []) or []
        duration_ms = int(max((s["end_sample"] for s in segs), default=0) * 1000 / sr)
        speaker_map = state.get("speaker_map", {}) or {}
        stable_ids = {
            s.get("speaker_id", speaker_map.get(s["cluster"], s["cluster"])) for s in segs
        }

        old = self.get_meta(mid) or {}
        meta = {
            "id": mid,
            "title": (state.get("title") or "Untitled Meeting").strip() or "Untitled Meeting",
            "created_at": old.get("created_at") or _now_iso(),
            "updated_at": _now_iso(),
            "sample_rate": sr,
            "merge_gap_s": float(state.get("merge_gap_s", 2.0)),
            "segments": len(segs),
            "duration_ms": duration_ms,
            "speakers": len(stable_ids),
        }

        payload = {"meta": meta, "state": dict(state)}
        tmp = d / "meeting.pkl.tmp"
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(tmp, d / "meeting.pkl")
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
                {"meta": meta, "transcript": {"paragraphs": paragraphs}},
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

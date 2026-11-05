"""Persist speaker identities with multiple voiceprint embeddings.

Storage: <SPEAKERS_DIR>/speakers.pkl, including float32 vectors.
Each embedding records its source, duration, meeting and segment references,
creation timestamp, and short flag. A short sample has a duration below
min_speaker_audio_s and may be less reliable for matching.
"""

from __future__ import annotations

import os
import pickle
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SpeakerStore:
    """Thread-safe pickle-backed speaker voiceprint library."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / "speakers.pkl"
        self._lock = threading.RLock()
        self._speakers: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------ helpers
    def valid_id(self, speaker_id: str) -> bool:
        return bool(_ID_RE.match(speaker_id))

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = pickle.loads(self._file.read_bytes())
                self._speakers = data.get("speakers", {}) or {}
            except Exception as exc:
                raise RuntimeError(
                    "Speaker library could not be read; refusing to erase existing identity records"
                ) from exc
        else:
            self._speakers = {}

    def _save(self) -> None:
        tmp = self._file.with_name("speakers-" + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_bytes(
                pickle.dumps(
                    {"speakers": self._speakers},
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
            os.replace(tmp, self._file)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _vector_meta(v: dict) -> dict:
        """Return voiceprint metadata without the embedding for JSON responses."""
        return {
            "id": v["id"],
            "duration_s": round(float(v.get("duration_s", 0.0)), 3),
            "source": v.get("source", "recording"),
            "meeting_id": v.get("meeting_id"),
            "segment_ids": list(v.get("segment_ids") or []),
            "text": v.get("text"),
            "created_at": v.get("created_at"),
            "short": bool(v.get("short", False)),
            "embedding_model": v.get("embedding_model", "camplus"),
            "human_verified": bool(v.get("human_verified", False)),
            "verification_source": v.get("verification_source"),
        }

    def _summary(self, s: dict) -> dict:
        return {
            "id": s["id"],
            "name": s["name"],
            "vector_count": len(s["vectors"]),
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "vectors": [self._vector_meta(v) for v in s["vectors"]],
        }

    # ------------------------------------------------------------ crud
    def list_speakers(self) -> list[dict]:
        with self._lock:
            out = [self._summary(s) for s in self._speakers.values()]
        out.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return out

    def create(self, name: str) -> dict:
        with self._lock:
            sid = uuid.uuid4().hex
            now = _now_iso()
            speaker = {
                "id": sid,
                "name": (name or "").strip() or "Unnamed Speaker",
                "created_at": now,
                "updated_at": now,
                "vectors": [],
            }
            self._speakers[sid] = speaker
            self._save()
            return self._summary(speaker)

    def rename(self, speaker_id: str, name: str) -> dict | None:
        with self._lock:
            s = self._speakers.get(speaker_id)
            if s is None:
                return None
            s["name"] = (name or "").strip() or "Unnamed Speaker"
            s["updated_at"] = _now_iso()
            self._save()
            return self._summary(s)

    def add_vectors(self, speaker_id: str, vectors: list[dict]) -> list[dict] | None:
        """Append embeddings and return their metadata, or None for unknown speakers."""
        with self._lock:
            s = self._speakers.get(speaker_id)
            if s is None:
                return None
            now = _now_iso()
            prepared: list[dict] = []
            for v in vectors:
                try:
                    emb = np.asarray(v["emb"], dtype=np.float32).reshape(-1)
                    duration = float(v.get("duration_s", 0.0))
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Invalid speaker embedding") from exc
                norm = float(np.linalg.norm(emb))
                if (not 32 <= emb.size <= 4096 or not np.isfinite(emb).all()
                        or not np.isfinite(norm) or norm <= 1e-8
                        or not np.isfinite(duration) or duration < 0):
                    raise ValueError("Invalid speaker embedding dimensions or values")
                model = v.get("embedding_model") or "camplus"
                if model not in ("camplus", "ecapa"):
                    raise ValueError("Unknown embedding-model provenance")
                verified = v.get("human_verified") is True
                meeting_id = v.get("meeting_id")
                segment_ids = list(v.get("segment_ids") or [])
                if verified and (
                    v.get("verification_source") != "manual_profile_correction"
                    or not isinstance(meeting_id, str) or not meeting_id
                    or not segment_ids or duration <= 0
                ):
                    raise ValueError("Verified enrollment requires manual source provenance")
                prepared.append({
                    "id": uuid.uuid4().hex,
                    "emb": np.ascontiguousarray(emb / norm),
                    "embedding_model": model,
                    "human_verified": verified,
                    "verification_source": (
                        "manual_profile_correction" if verified else None
                    ),
                    "duration_s": duration,
                    "source": v.get("source", "recording"),
                    "meeting_id": meeting_id,
                    "segment_ids": segment_ids,
                    "text": v.get("text"),
                    "created_at": now,
                    "short": bool(v.get("short", False)),
                })
            prior_updated_at = s["updated_at"]
            s["vectors"].extend(prepared)
            s["updated_at"] = now
            try:
                self._save()
            except Exception:
                del s["vectors"][-len(prepared):]
                s["updated_at"] = prior_updated_at
                raise
            return [self._vector_meta(row) for row in prepared]

    def delete_vector(self, speaker_id: str, vector_id: str) -> bool:
        with self._lock:
            s = self._speakers.get(speaker_id)
            if s is None:
                return False
            for i, v in enumerate(s["vectors"]):
                if v["id"] == vector_id:
                    del s["vectors"][i]
                    s["updated_at"] = _now_iso()
                    self._save()
                    return True
            return False

    def delete_speaker(self, speaker_id: str) -> bool:
        with self._lock:
            if speaker_id not in self._speakers:
                return False
            del self._speakers[speaker_id]
            self._save()
            return True

    # ------------------------------------------------------------ matching
    def get_for_matching(self, speaker_ids: list[str]) -> list[dict]:
        """Return full speaker records with embeddings for participant matching."""
        with self._lock:
            out: list[dict] = []
            for sid in speaker_ids:
                s = self._speakers.get(sid)
                if not s:
                    continue
                out.append(
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "vectors": [dict(v) for v in s["vectors"]],
                    }
                )
            return out

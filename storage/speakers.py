"""说话人声纹库持久化：每个说话人可以保存多条声纹特征向量。

存储结构：`<SPEAKERS_DIR>/speakers.pkl`（pickle 单文件，含 float32 声纹向量）。
每条声纹向量记录来源（现场录音 / 会议片段）、时长、关联会议与片段、
创建时间，以及 `short` 标记（时长低于配置的 `min_speaker_audio_s`，
提示该语音过短、声纹特征可能不准确）。
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
    """说话人声纹库，线程安全，pickle 持久化。"""

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
            except Exception:
                self._speakers = {}
        else:
            self._speakers = {}

    def _save(self) -> None:
        tmp = self._file.with_name("speakers.pkl.tmp")
        tmp.write_bytes(
            pickle.dumps({"speakers": self._speakers}, protocol=pickle.HIGHEST_PROTOCOL)
        )
        os.replace(tmp, self._file)

    @staticmethod
    def _vector_meta(v: dict) -> dict:
        """API 使用的声纹元数据（不含 emb，便于 JSON 序列化）。"""
        return {
            "id": v["id"],
            "duration_s": round(float(v.get("duration_s", 0.0)), 3),
            "source": v.get("source", "recording"),
            "meeting_id": v.get("meeting_id"),
            "segment_ids": list(v.get("segment_ids") or []),
            "text": v.get("text"),
            "created_at": v.get("created_at"),
            "short": bool(v.get("short", False)),
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
                "name": (name or "").strip() or "未命名说话人",
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
            s["name"] = (name or "").strip() or "未命名说话人"
            s["updated_at"] = _now_iso()
            self._save()
            return self._summary(s)

    def add_vectors(self, speaker_id: str, vectors: list[dict]) -> list[dict] | None:
        """追加声纹向量，返回新增向量的元数据；说话人不存在时返回 None。"""
        with self._lock:
            s = self._speakers.get(speaker_id)
            if s is None:
                return None
            now = _now_iso()
            added: list[dict] = []
            for v in vectors:
                vec = {
                    "id": uuid.uuid4().hex,
                    "emb": np.asarray(v["emb"], dtype=np.float32).reshape(-1),
                    "duration_s": float(v.get("duration_s", 0.0)),
                    "source": v.get("source", "recording"),
                    "meeting_id": v.get("meeting_id"),
                    "segment_ids": list(v.get("segment_ids") or []),
                    "text": v.get("text"),
                    "created_at": now,
                    "short": bool(v.get("short", False)),
                }
                s["vectors"].append(vec)
                added.append(self._vector_meta(vec))
            s["updated_at"] = now
            self._save()
            return added

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
        """供会议处理器做参与者匹配：返回含 emb 数组的完整说话人快照。"""
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

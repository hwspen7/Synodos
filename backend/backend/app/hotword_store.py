"""热词库持久化：每个热词库保存一组热词，会议开始时按库选择后注入 ASR。

存储结构：`<HOTWORDS_DIR>/hotwords.pkl`（pickle 单文件）。
每个热词库包含名称、热词列表（自动去重、忽略空串）与创建/更新时间。
"""

from __future__ import annotations

import os
import pickle
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_words(words) -> list[str]:
    """清洗热词：去空白、去重、忽略空串，保留顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in words or []:
        if not isinstance(item, str):
            continue
        for w in item.replace("，", " ").replace("、", " ").replace("；", " ").split():
            w = w.strip()
            if w and w not in seen:
                seen.add(w)
                out.append(w)
    return out


class HotwordStore:
    """热词库，线程安全，pickle 持久化。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / "hotwords.pkl"
        self._lock = threading.RLock()
        self._libraries: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------ helpers
    def valid_id(self, hotword_id: str) -> bool:
        return bool(_ID_RE.match(hotword_id))

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = pickle.loads(self._file.read_bytes())
                self._libraries = data.get("libraries", {}) or {}
            except Exception:
                self._libraries = {}
        else:
            self._libraries = {}

    def _save(self) -> None:
        tmp = self._file.with_name("hotwords.pkl.tmp")
        tmp.write_bytes(
            pickle.dumps({"libraries": self._libraries}, protocol=pickle.HIGHEST_PROTOCOL)
        )
        os.replace(tmp, self._file)

    def _summary(self, lib: dict) -> dict:
        return {
            "id": lib["id"],
            "name": lib["name"],
            "words": list(lib["words"]),
            "word_count": len(lib["words"]),
            "created_at": lib["created_at"],
            "updated_at": lib["updated_at"],
        }

    # ------------------------------------------------------------ crud
    def list_hotwords(self) -> list[dict]:
        with self._lock:
            out = [self._summary(lib) for lib in self._libraries.values()]
        out.sort(key=lambda lib: lib.get("updated_at", ""), reverse=True)
        return out

    def create(self, name: str, words=None) -> dict:
        with self._lock:
            lib_id = uuid.uuid4().hex
            now = _now_iso()
            lib = {
                "id": lib_id,
                "name": (name or "").strip() or "未命名热词库",
                "words": normalize_words(words),
                "created_at": now,
                "updated_at": now,
            }
            self._libraries[lib_id] = lib
            self._save()
            return self._summary(lib)

    def update(self, hotword_id: str, name=None, words=None) -> dict | None:
        """更新热词库名称与热词列表；未提供的字段保持不变。"""
        with self._lock:
            lib = self._libraries.get(hotword_id)
            if lib is None:
                return None
            if name is not None:
                lib["name"] = (name or "").strip() or "未命名热词库"
            if words is not None:
                lib["words"] = normalize_words(words)
            lib["updated_at"] = _now_iso()
            self._save()
            return self._summary(lib)

    def delete(self, hotword_id: str) -> bool:
        with self._lock:
            if hotword_id not in self._libraries:
                return False
            del self._libraries[hotword_id]
            self._save()
            return True

    # ------------------------------------------------------------ meeting
    def get_words_for_meeting(self, hotword_ids: list[str], limit: int = 200) -> list[str]:
        """按库 ID 取热词并合并去重，供 ASR 注入使用。"""
        with self._lock:
            out: list[str] = []
            seen: set[str] = set()
            for hid in hotword_ids:
                lib = self._libraries.get(hid)
                if not lib:
                    continue
                for w in lib["words"]:
                    if w in seen:
                        continue
                    seen.add(w)
                    out.append(w)
                    if len(out) >= limit:
                        break
                if len(out) >= limit:
                    break
        return out

    def get_summaries(self, hotword_ids: list[str]) -> list[dict]:
        """返回指定库的摘要（不含 words 全文），供会议 ready 消息使用。"""
        with self._lock:
            return [
                {
                    "id": lib["id"],
                    "name": lib["name"],
                    "word_count": len(lib["words"]),
                }
                for hid in hotword_ids
                if (lib := self._libraries.get(hid)) is not None
            ]

"""Persist named hotword libraries for selection at meeting startup.

Storage format: <HOTWORDS_DIR>/hotwords.pkl (a single pickle file).
Each library stores a name, deduplicated words, and creation/update timestamps.
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
    """Trim whitespace, skip empty entries, deduplicate, and preserve order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in words or []:
        if not isinstance(item, str):
            continue
        for w in item.replace("\uFF0C", " ").replace("\u3001", " ").replace("\uFF1B", " ").split():
            w = w.strip()
            if w and w not in seen:
                seen.add(w)
                out.append(w)
    return out


class HotwordStore:
    """Thread-safe pickle-backed hotword library."""

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
                "name": (name or "").strip() or "Untitled Hotword Library",
                "words": normalize_words(words),
                "created_at": now,
                "updated_at": now,
            }
            self._libraries[lib_id] = lib
            self._save()
            return self._summary(lib)

    def update(self, hotword_id: str, name=None, words=None) -> dict | None:
        """Update a hotword library while preserving unspecified fields."""
        with self._lock:
            lib = self._libraries.get(hotword_id)
            if lib is None:
                return None
            if name is not None:
                lib["name"] = (name or "").strip() or "Untitled Hotword Library"
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
        """Merge hotwords from selected library IDs without duplicates for ASR."""
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
        """Return selected library summaries without their full word lists."""
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

"""每个 WebSocket 会话的后台处理线程。

处理流程：
1. 缓冲音频累积超过 process_threshold（默认 10s）后，对整段缓冲做一次 ASR
   （paraformer + fsmn-vad + ct-punc，sentence_timestamp=True，内部自带 VAD 切句）
2. ASR 返回的句子中，结束点距缓冲末尾超过 trusted_threshold （默认 2s） 的句子视为可信段：
   - 该句文本一经确认不再重算（不会因为后续聚类/合并结果变化而重新 ASR）
   - 用 cam++ 从该句音频提取一次声纹向量，不再重算
3. 说话人标签计算：
   - 对每个可信段，先与每个被选择的“会议参与者”的全部声纹向量做余弦相似度计算，
     取最大值；若超过阈值（默认 0.5，前端可调），直接标注为该参与者的姓名；
   - 其余可信段交给 FunASR ClusterBackend 全量重聚类，
     分配“未知说话人1 / 未知说话人2 …”标签（未选择参与者时保持“说话人N”），
     并把原始簇标签映射为稳定说话人 ID，减少前端标签抖动
4. 可信段列表有变化时，把完整 segment 快照推给前端；
   段落（paragraph）不再由后端计算，改由前端按“同说话人 + 间隔 < merge_gap”展示

每个可信段只确认一次：确认后立即从窗口裁剪掉，之后的任何聚类变化
都不会再碰它的文本与声纹，因此更省算力、结果更稳定。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import numpy as np

from .buffer import AudioBuffer
from .config import Settings
from .hotword_store import HotwordStore
from .models import ModelManager
from .speaker_store import SpeakerStore
from .store import MeetingStore

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
        # 注意：不能用 self._stop 命名——会遮蔽 threading.Thread._stop()，
        # 导致 join() 内部调用 _stop() 时报 "'Event' object is not callable"
        self._stop_event = stop_event or threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        sr = settings.sample_rate
        # 缓冲达到该时长触发一轮 ASR 处理
        self._process_threshold_s = settings.vad_threshold_s
        # 句子结束点距缓冲末尾超过该时长才视为可信段
        self._trusted_threshold_s = settings.rec_threshold_s
        # 收尾时缓冲剩余少于该时长直接丢弃
        self._final_discard_s = settings.final_discard_s

        # 全局可信段
        self.segments: list[dict] = []
        self._next_segment_id = 1

        # 原始簇标签 -> 稳定说话人 ID 的映射
        self._speaker_map: dict[int, int] = {}
        self._speaker_seg_ids: dict[int, set[int]] = {}

        self._last_fingerprint: Optional[tuple] = None
        self._transcript_version = 0
        self._last_state_sent = 0.0

        # 会议持久化（保存/续记）
        self.meeting_store = meeting_store
        self.meeting_id = meeting_id
        self.meeting_title = meeting_title
        self._save_requested = threading.Event()
        self._save_done = threading.Event()

        # 会议参与者声纹匹配（说话人声纹库）
        self.speaker_store = speaker_store
        self.participant_ids: list[str] = list(participant_ids or [])
        # 会议热词库（热词识别）：开会时按库选择，注入 ASR hotword 参数
        self.hotword_store: Optional[HotwordStore] = hotword_store
        self.hotword_ids: list[str] = list(hotword_ids or [])
        self._hotword_str: str = ""
        # 单次会话最多注入的热词数，防止热词过多影响识别速度/效果
        self._hotword_limit = 200
        self.spk_threshold: float = float(
            spk_threshold
            if spk_threshold is not None
            else settings.spk_match_threshold
        )
        self._participants: list[dict] = []
        self._participant_label_map: dict[str, int] = {}
        # 未知说话人展示序号（与参与者标签独立，始终从 1 开始）
        self._unknown_index: dict[int, int] = {}
        # 前端是否显式传入了参与者/阈值；未传入时允许从会议保存的状态恢复
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
                    # 本轮没有新内容：等缓冲区再增长约 0.5s 再检查，避免空转
                    current = self.buffer.seconds()
                    self.buffer.wait_until(current + 0.5, self._stop_event, self._save_requested)
            # 收尾：把窗口里剩余（尚未满足确认阈值）的语音也处理掉
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
        # 收尾时自动保存到磁盘
        self._do_save()
        self._save_done.set()

    # ------------------------------------------------------------ persistence
    def request_save(self, timeout: float = 30.0) -> bool:
        """请求处理器线程保存会议状态（由保存接口调用）。"""
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
        """导出可落盘的完整会话状态（处理器线程内调用）。"""
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
                    "emb": s["emb"],  # float32 数组，pickle 直接序列化更省空间
                }
                for s in self.segments
            ],
            "next_segment_id": self._next_segment_id,
            "speaker_map": dict(self._speaker_map),
        }

    def load_state(self, state: dict) -> None:
        """恢复已保存的会议状态，从末尾继续记录。"""
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

        # 不保存原始音频：续记时从最后一个已确认片段的末尾继续
        offset = max((int(s["end_sample"]) for s in self.segments), default=0)
        self.buffer.set_start_offset(offset)

        self._last_fingerprint = None
        self.meeting_title = state.get("title") or self.meeting_title

    def load_participants(self) -> None:
        """从声纹库加载被选择的会议参与者（含全部声纹向量）。"""
        if not self.speaker_store or not self.participant_ids:
            self._participants = []
            return
        self._participants = self.speaker_store.get_for_matching(self.participant_ids)
        # 参与者可能已被删除，同步过滤
        self.participant_ids = [p["id"] for p in self._participants]

    def load_hotwords(self) -> None:
        """从热词库加载被选择的库，合并成空格分隔的热词串注入 ASR。"""
        if not self.hotword_store or not self.hotword_ids:
            self._hotword_str = ""
            return
        words = self.hotword_store.get_words_for_meeting(
            self.hotword_ids, limit=self._hotword_limit
        )
        # 热词库可能已被删除，只保留仍然存在的库
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
        """恢复会议后按当前参与者/阈值重新计算一遍说话人标签。"""
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

        # 对整段缓冲做一次 ASR（内部 fsmn-vad 切句 + 标点恢复）
        sentences = self.models.transcribe_sentences(audio, hotword=self._hotword_str or None)
        local_end = 0
        for sentence in sentences:
            start_ms = int(sentence["start"])
            end_ms = int(sentence["end"])
            text = sentence["text"]
            # 可信条件：句子结束距缓冲末尾足够远（收尾时不再要求）
            if not final and buffer_s - end_ms / 1000.0 < self._trusted_threshold_s:
                break

            s_g = start_offset + int(start_ms * sr / 1000)
            e_g = start_offset + int(end_ms * sr / 1000)
            if e_g <= s_g:
                continue

            # 从当前缓冲切片提取声纹
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

        # 裁剪掉已确认的音频（新来的音频不受影响）
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
        """计算所有可信段的说话人标签。

        1. 每个段先与所选会议参与者的全部声纹做余弦匹配，超过阈值直接标注为参与者姓名；
        2. 其余段交给 FunASR ClusterBackend 聚类，分配“未知说话人N”；
        3. 把原始簇标签映射为稳定说话人 ID，减少标签抖动。
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

        # 剩余段进入 funasr 默认聚类
        if pool:
            embs = np.stack([self.segments[i]["emb"] for i in pool]).astype(np.float32)
            labels = self.models.cluster(embs)
            for idx, label in zip(pool, labels):
                self.segments[idx]["cluster"] = int(label)

        self._remap_speakers()
        self._assign_labels(matched)

    def _match_participant(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """与每个参与者的每条声纹做余弦相似度，返回 (参与者id, 最大相似度)。"""
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
        """只对进入聚类的段（cluster >= 0）做稳定映射，跳过参与者段。"""
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
                # 复用被释放的 ID，避免说话人 ID 随着聚类变化不断增大（ID 从 1 开始）
                free_id = self._next_free_speaker_id(used)
                new_map[raw] = free_id
                used.add(free_id)
        self._speaker_map = new_map
        self._speaker_seg_ids = new_seg_ids

    def _assign_labels(self, matched: dict[int, str]) -> None:
        """给每个段写入稳定的 speaker_id 与展示用的 speaker_name。"""
        for participant_id in matched.values():
            if participant_id not in self._participant_label_map:
                self._participant_label_map[participant_id] = self._next_free_speaker_id()
        names = {p["id"]: p["name"] for p in self._participants}
        for seg in self.segments:
            if seg["cluster"] == -1:
                participant_id = matched.get(seg["id"])
                stable = self._participant_label_map.get(participant_id or "", 1)
                seg["speaker_id"] = stable
                seg["speaker_name"] = names.get(participant_id or "", "参与者")
            else:
                stable = self._speaker_map.get(seg["cluster"], seg["cluster"])
                seg["speaker_id"] = stable
                if self.participant_ids:
                    if stable not in self._unknown_index:
                        self._unknown_index[stable] = (
                            max(self._unknown_index.values(), default=0) + 1
                        )
                    seg["speaker_name"] = f"未知说话人{self._unknown_index[stable]}"
                else:
                    seg["speaker_name"] = f"说话人{stable}"

    # ---------------------------------------------------------------- push
    def _rebuild_and_push(self) -> bool:
        """重建 segment 快照并推送给前端；内容没变化时不推送。"""
        ordered = sorted(self.segments, key=lambda s: s["start_sample"])
        items = [
            {
                "id": s["id"],
                "speaker_id": s.get(
                    "speaker_id", self._speaker_map.get(s["cluster"], s["cluster"])
                ),
                "speaker_name": s.get("speaker_name")
                or f"说话人{s.get('speaker_id', self._speaker_map.get(s['cluster'], s['cluster']))}",
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

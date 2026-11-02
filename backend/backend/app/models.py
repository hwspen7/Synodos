"""ASR / 声纹 / 聚类模型的管理器（惰性加载 + 串行推理）。

- paraformer-zh + fsmn-vad + ct-punc-c 做 ASR（sentence_timestamp=True，内部切句 + 标点恢复）
- cam++ 提取 192 维声纹 embedding
- FunASR ClusterBackend（谱聚类/UMAP+HDBSCAN + 余弦合并）做全量重聚类

funasr 的 generate 不是线程安全的，所有推理用全局锁串行化。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


def clean_asr_text(text: str) -> str:
    """去掉 paraformer 输出中汉字之间的空格，保留标点与英文单词。"""
    if not text:
        return ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


class ModelManager:
    """共享的模型单例。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._asr = None
        self._spk = None
        self._cluster = None

    # ------------------------------------------------------------------ asr
    def _load_asr(self):
        with self._load_lock:
            if self._asr is None:
                from funasr import AutoModel

                t0 = time.time()
                logger.info(
                    "Loading ASR model %s + fsmn-vad %s + punc %s ...",
                    self.settings.asr_model,
                    self.settings.fsmn_vad_model,
                    self.settings.punc_model,
                )
                self._asr = AutoModel(
                    model=self.settings.asr_model,
                    vad_model=self.settings.fsmn_vad_model,
                    punc_model=self.settings.punc_model,
                    disable_log=True,
                    disable_update=True,
                    ncpu=os.cpu_count() or 4,
                )
                logger.info("ASR model loaded in %.1fs", time.time() - t0)
            return self._asr

    def transcribe(self, samples: np.ndarray, hotword: str | None = None) -> str:
        """对一段 16k 单声道 float32 音频做 ASR（fsmn-vad 内部分段 + 标点恢复）。"""
        if samples is None or len(samples) == 0:
            return ""
        with self._infer_lock:
            res = self._load_asr().generate(
                input=samples,
                hotword=hotword or None,
                cache={},
                disable_pbar=True,
            )
        text = ""
        if res and isinstance(res[0], dict):
            text = res[0].get("text", "") or ""
        return clean_asr_text(text)

    def transcribe_sentences(
        self, samples: np.ndarray, hotword: str | None = None
    ) -> list[dict]:
        """对一段 16k 单声道 float32 音频做 ASR，返回带时间戳的句子列表。

        返回 [{"start": ms, "end": ms, "text": str}, ...]。
        hotword 为空格分隔的热词字符串（如 "达摩院 魔搭"），为空时不启用热词。
        """
        if samples is None or len(samples) == 0:
            return []
        with self._infer_lock:
            res = self._load_asr().generate(
                input=samples,
                hotword=hotword or None,
                sentence_timestamp=True,
                cache={},
                disable_pbar=True,
            )
        if not res or not isinstance(res[0], dict):
            return []
        info = res[0].get("sentence_info") or []
        out: list[dict] = []
        for item in info:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "start": int(item.get("start", 0)),
                    "end": int(item.get("end", 0)),
                    "text": clean_asr_text(item.get("text", "") or ""),
                }
            )
        return out

    # ------------------------------------------------------------------ spk
    def _load_spk(self):
        with self._load_lock:
            if self._spk is None:
                from funasr import AutoModel

                t0 = time.time()
                logger.info("Loading speaker model %s ...", self.settings.spk_model)
                self._spk = AutoModel(
                    model=self.settings.spk_model,
                    disable_log=True,
                    disable_update=True,
                    ncpu=os.cpu_count() or 4,
                )
                logger.info("Speaker model loaded in %.1fs", time.time() - t0)
            return self._spk

    def embed(self, samples: np.ndarray) -> np.ndarray | None:
        """提取一段音频的声纹向量，L2 归一化后的 1-D float32（如 192 维）。"""
        if samples is None or len(samples) == 0:
            return None
        with self._infer_lock:
            res = self._load_spk().generate(
                input=samples,
                cache={},
                disable_pbar=True,
            )
        if not res or not isinstance(res[0], dict):
            return None
        emb = res[0].get("spk_embedding")
        if emb is None:
            return None
        vec = np.asarray(emb.detach().cpu().numpy()).reshape(-1).astype("float32")
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    # --------------------------------------------------------------- cluster
    def _load_cluster(self):
        with self._load_lock:
            if self._cluster is None:
                from funasr.models.campplus.cluster_backend import ClusterBackend

                t0 = time.time()
                logger.info("Loading ClusterBackend (merge_thr=%.2f) ...", self.settings.cluster_merge_thr)
                self._cluster = ClusterBackend(merge_thr=self.settings.cluster_merge_thr)
                logger.info("ClusterBackend loaded in %.1fs", time.time() - t0)
            return self._cluster

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """对 [N, D] 的声纹矩阵做全量聚类，返回长度 N 的 int 标签数组。"""
        if embeddings is None or len(embeddings) == 0:
            return np.empty(0, dtype=int)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a [N, D] matrix")
        with self._infer_lock:
            labels = self._load_cluster()(embeddings, oracle_num=None)
        return np.asarray(labels).reshape(-1).astype(int)

    # ------------------------------------------------------------------ misc
    def loaded_state(self) -> dict[str, bool]:
        return {
            "asr": self._asr is not None,
            "spk": self._spk is not None,
            "cluster": self._cluster is not None,
        }

    def preload(self) -> None:
        logger.info("Preloading models ...")
        self._load_asr()
        self._load_spk()
        self._load_cluster()
        logger.info("All models preloaded.")

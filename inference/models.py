"""Model manager for ASR, voiceprints, and clustering (lazy loading).

- Paraformer, FSMN-VAD, and CT-Transformer provide ASR, segmentation, and punctuation.
- CAM++ extracts 192-dimensional speaker embeddings.
- FunASR ClusterBackend reclusters all unmatched speaker embeddings.

FunASR generate calls share a global inference lock for thread safety.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import numpy as np

from core.config import Settings
from foundation.settings import InfrastructureSettings, load_infrastructure_settings
from inference.scheduler import InferenceScheduler

logger = logging.getLogger(__name__)


def clean_asr_text(text: str) -> str:
    """Remove redundant spaces between Chinese characters in ASR output."""
    if not text:
        return ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


class ModelManager:
    """Shared singleton model manager."""

    def __init__(self, settings: Settings, infrastructure: InfrastructureSettings | None = None):
        self.settings = settings
        limits = infrastructure or load_infrastructure_settings()
        self._scheduler = InferenceScheduler(
            max_queue=limits.inference_queue_size,
            timeout_s=limits.inference_timeout_s,
        )
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
        return self._scheduler.submit("asr", lambda: self._transcribe(samples, hotword), priority=0)

    def _transcribe(self, samples: np.ndarray, hotword: str | None = None) -> str:
        """Run ASR on 16 kHz mono float32 audio with VAD and punctuation."""
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
        return self._scheduler.submit(
            "asr_sentences", lambda: self._transcribe_sentences(samples, hotword), priority=0
        )

    def _transcribe_sentences(
        self, samples: np.ndarray, hotword: str | None = None
    ) -> list[dict]:
        """Transcribe a 16 kHz mono float32 audio window into timed sentences.

        Return a list of dictionaries with start/end milliseconds and recognized text.
        Hotwords are separated by spaces; None disables hotword biasing.
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
        return self._scheduler.submit("embedding", lambda: self._embed(samples), priority=5)

    def _embed(self, samples: np.ndarray) -> np.ndarray | None:
        """Return an L2-normalized one-dimensional float32 speaker embedding."""
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
        return self._scheduler.submit("clustering", lambda: self._cluster_embeddings(embeddings), priority=10)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster an [N, D] embedding matrix and return N integer labels."""
        if embeddings is None or len(embeddings) == 0:
            return np.empty(0, dtype=int)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a [N, D] matrix")
        with self._infer_lock:
            labels = self._load_cluster()(embeddings, oracle_num=None)
        return np.asarray(labels).reshape(-1).astype(int)

    # ------------------------------------------------------- multilingual ASR
    def recognize_multilingual(self, samples: np.ndarray, *, language: str = "auto",
                               terms: str = "", config=None):
        """Run multilingual recognition through the shared inference scheduler.

        Keep scheduler and inference-lock ownership inside ModelManager rather
        than requiring processors to access private implementation details.
        """
        from speech.engine import recognize

        def inference_job():
            with self._infer_lock:
                return recognize(samples, language=language, terms=terms, config=config)

        return self._scheduler.submit("multilingual_asr", inference_job, priority=0)

    # ------------------------------------------------------------------ misc
    def loaded_state(self) -> dict[str, bool]:
        return {
            "asr": self._asr is not None,
            "spk": self._spk is not None,
            "cluster": self._cluster is not None,
        }

    def preload(self) -> None:
        logger.info("Preloading models ...")
        from speech.config import load_settings
        speech_config = load_settings()
        if speech_config.engine == "legacy":
            self._load_asr()
        elif os.getenv("MEETING_PREWARM_ASR", "1").strip().lower() in ("1", "true", "yes"):
            from speech.engine import prewarm_multilingual
            prewarm_multilingual(speech_config)
        self._load_spk()
        self._load_cluster()
        logger.info("All models preloaded.")

    def close(self) -> None:
        self._scheduler.close()

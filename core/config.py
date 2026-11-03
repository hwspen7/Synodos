"""Configuration for the realtime meeting backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


SAMPLE_RATE = 16000


# FunASR model aliases mapped to ModelScope repository IDs.
# Resolve version-dependent aliases into repository or local cache paths.
FUNASR_MODEL_MAP = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "ct-punc-c": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
}


def resolve_model(name: str) -> str:
    """Prefer an existing local model cache to repeated network lookups.

    Accepted model identifiers:
    1. An existing local directory, used as-is.
    2. A known FunASR alias, resolved through the ModelScope cache when available.
    3. Any other identifier, passed to FunASR for resolution or download.
    """
    if Path(name).exists():
        return name
    target = FUNASR_MODEL_MAP.get(name, name)
    cache_root = Path(
        os.environ.get(
            "MODELSCOPE_CACHE",
            Path.home() / ".cache" / "modelscope" / "hub" / "models",
        )
    )
    local = cache_root / target
    if local.exists():
        return str(local)
    return target


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    sample_rate: int = SAMPLE_RATE

    #: Trigger ASR when buffered audio reaches this duration in seconds.
    vad_threshold_s: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD_S", 10.0))
    #: Treat a sentence as confirmed after this much audio follows its ending.
    #: Confirmation prevents repeated ASR for the finalized sentence.
    rec_threshold_s: float = field(default_factory=lambda: _env_float("REC_THRESHOLD_S", 2.0))
    #: Merge adjacent segments of one speaker when the gap is below this threshold.
    merge_gap_s: float = field(default_factory=lambda: _env_float("MERGE_GAP_S", 2.0))
    #: Legacy compatibility only. Final audio is always submitted to ASR.
    final_discard_s: float = field(default_factory=lambda: _env_float("FINAL_DISCARD_S", 0.0))
    #: Directory for meeting records.
    meetings_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MEETINGS_DIR", "meetings"))
    )
    #: Directory for saved speaker voiceprints.
    speakers_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SPEAKERS_DIR", "speakers"))
    )
    #: Directory for hotword libraries.
    hotwords_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("HOTWORDS_DIR", "hotwords"))
    )

    #: Cosine-similarity merge threshold for the FunASR clustering backend.
    cluster_merge_thr: float = field(
        default_factory=lambda: _env_float("CLUSTER_MERGE_THR", 0.78)
    )
    #: Warn when a speaker sample is shorter than this duration in seconds.
    min_speaker_audio_s: float = field(
        default_factory=lambda: _env_float("MIN_SPEAKER_AUDIO_S", 5.0)
    )
    #: Default cosine-similarity threshold for matching selected participants.
    spk_match_threshold: float = field(
        default_factory=lambda: _env_float("SPK_MATCH_THRESHOLD", 0.3)
    )

    asr_model: str = field(default_factory=lambda: resolve_model(_env_str("ASR_MODEL", "paraformer-zh")))
    fsmn_vad_model: str = field(default_factory=lambda: resolve_model(_env_str("FSMN_VAD_MODEL", "fsmn-vad")))
    punc_model: str = field(default_factory=lambda: resolve_model(_env_str("PUNC_MODEL", "ct-punc-c")))
    spk_model: str = field(default_factory=lambda: resolve_model(_env_str("SPK_MODEL", "cam++")))


def make_settings() -> Settings:
    return Settings()

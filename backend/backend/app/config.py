"""Configuration for the realtime meeting backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


SAMPLE_RATE = 16000


# funasr 的短名 -> ModelScope 仓库路径。
# 短名在不同 funasr 版本里注册情况不一致，这里统一解析成仓库路径/本地缓存路径。
FUNASR_MODEL_MAP = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "ct-punc-c": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
}


def resolve_model(name: str) -> str:
    """优先使用本地已下载的模型缓存，避免每次启动都走网络/写缓存。

    传入的 name 可以是：
    1. 本地目录路径（直接使用）
    2. funasr 短名（如 paraformer-zh）：若 ModelScope 缓存目录中已有对应模型，则返回缓存路径
    3. 其他情况原样返回，交由 funasr 自行下载
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

    #: 缓冲区音频达到该时长（秒）时触发一轮 ASR 处理
    vad_threshold_s: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD_S", 10.0))
    #: ASR 句子结束点距缓冲区末尾达到该时长（秒）才视为“可信段”，
    #: 即确认该句已说完
    rec_threshold_s: float = field(default_factory=lambda: _env_float("REC_THRESHOLD_S", 2.0))
    #: 同一说话人的相邻可信段间隔小于该时长（秒）时合并为一个段落（空隙一起带进去做 ASR）
    merge_gap_s: float = field(default_factory=lambda: _env_float("MERGE_GAP_S", 2.0))
    #: 收尾时缓冲剩余少于该时长（秒）直接丢弃
    final_discard_s: float = field(default_factory=lambda: _env_float("FINAL_DISCARD_S", 1.0))
    #: 会议记录保存目录
    meetings_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MEETINGS_DIR", "meetings"))
    )
    #: 说话人声纹库保存目录
    speakers_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SPEAKERS_DIR", "speakers"))
    )
    #: 热词库保存目录
    hotwords_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("HOTWORDS_DIR", "hotwords"))
    )

    #: FunASR ClusterBackend 的余弦相似度合并阈值
    cluster_merge_thr: float = field(
        default_factory=lambda: _env_float("CLUSTER_MERGE_THR", 0.78)
    )
    #: 说话人录音/片段低于该时长（秒）时提示“语音过短，声纹可能不准确”
    min_speaker_audio_s: float = field(
        default_factory=lambda: _env_float("MIN_SPEAKER_AUDIO_S", 5.0)
    )
    #: 会议参与者声纹匹配的默认余弦相似度阈值（前端可覆盖）
    spk_match_threshold: float = field(
        default_factory=lambda: _env_float("SPK_MATCH_THRESHOLD", 0.3)
    )

    asr_model: str = field(default_factory=lambda: resolve_model(_env_str("ASR_MODEL", "paraformer-zh")))
    fsmn_vad_model: str = field(default_factory=lambda: resolve_model(_env_str("FSMN_VAD_MODEL", "fsmn-vad")))
    punc_model: str = field(default_factory=lambda: resolve_model(_env_str("PUNC_MODEL", "ct-punc-c")))
    spk_model: str = field(default_factory=lambda: resolve_model(_env_str("SPK_MODEL", "cam++")))


def make_settings() -> Settings:
    return Settings()

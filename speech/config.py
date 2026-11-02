"""Typed, validated multilingual speech settings."""
from __future__ import annotations
from dataclasses import dataclass
import os

SUPPORTED_ENGINES = {"legacy", "whisper", "qwen3"}


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"{name} must be a boolean")
    return value in ("true", "1", "yes")


@dataclass(frozen=True)
class SpeechSettings:
    engine: str = "legacy"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    preview_interval_s: float = 2.5
    confirmation_delay_s: float = 1.0
    language: str = "auto"
    translate_to: str = ""
    record_audio: bool = False
    enable_events: bool = False
    enable_overlap: bool = False
    enable_denoise: bool = False
    native_streaming: bool = False
    recording_root: str = "recordings"
    max_recording_bytes: int = 4_000_000_000

    def validate(self) -> "SpeechSettings":
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError("MEETING_ASR_ENGINE must be legacy, whisper, or qwen3")
        if not 0.5 <= self.preview_interval_s <= 30:
            raise ValueError("MEETING_PREVIEW_INTERVAL_S must be between 0.5 and 30")
        if not 0.2 <= self.confirmation_delay_s <= 15:
            raise ValueError("MEETING_CONFIRMATION_DELAY_S must be between 0.2 and 15")
        if self.native_streaming and self.engine == "legacy":
            raise ValueError("Native streaming requires a multilingual engine")
        if not 1_000_000 <= self.max_recording_bytes <= 100_000_000_000:
            raise ValueError("MEETING_MAX_RECORDING_BYTES out of range")
        return self


def load_settings() -> SpeechSettings:
    return SpeechSettings(
        engine=os.getenv("MEETING_ASR_ENGINE", "legacy").lower(),
        model=os.getenv("MEETING_ASR_MODEL", "small"),
        device=os.getenv("MEETING_ASR_DEVICE", "auto"),
        compute_type=os.getenv("MEETING_ASR_COMPUTE", "default"),
        preview_interval_s=float(os.getenv("MEETING_PREVIEW_INTERVAL_S", "2.5")),
        confirmation_delay_s=float(os.getenv("MEETING_CONFIRMATION_DELAY_S", "1")),
        language=os.getenv("MEETING_ASR_LANGUAGE", "auto"),
        translate_to=os.getenv("MEETING_TRANSLATE_TO", ""),
        record_audio=_flag("MEETING_RECORD_AUDIO"),
        enable_events=_flag("MEETING_AUDIO_EVENTS"),
        enable_overlap=_flag("MEETING_OVERLAP_DETECTION"),
        enable_denoise=_flag("MEETING_DENOISE"),
        native_streaming=_flag("MEETING_NATIVE_STREAMING"),
        recording_root=os.getenv("MEETING_RECORDING_ROOT", "recordings"),
        max_recording_bytes=int(os.getenv("MEETING_MAX_RECORDING_BYTES", "4000000000")),
    ).validate()

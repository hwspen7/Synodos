"""Opt-in local intelligence models and storage settings."""
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def _flag(name: str, default=False) -> bool:
    return os.getenv(name, str(int(default))).lower() in ('1', 'yes', 'true', 'on')


@dataclass(frozen=True)
class AIConfig:
    enabled: bool
    provider: str
    endpoint: str
    text_model: str
    embed_model: str
    vision_model: str
    api_key: str
    media_root: Path
    max_upload_bytes: int
    timeout_s: int
    enable_embeddings: bool
    enable_vision: bool
    enable_audio_import: bool


def load_config() -> AIConfig:
    provider = os.getenv('MEETING_AI_PROVIDER', 'ollama').strip().lower()
    if provider not in ('ollama', 'openai_compatible'):
        raise ValueError('Unsupported MEETING_AI_PROVIDER')
    endpoint = os.getenv('MEETING_AI_ENDPOINT', 'http://127.0.0.1:11434').rstrip('/')
    parsed = urlparse(endpoint)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Invalid AI endpoint')
    if parsed.hostname not in ('localhost', '127.0.0.1', '::1') and not _flag('MEETING_AI_ALLOW_REMOTE'):
        raise ValueError('Remote AI endpoints require explicit MEETING_AI_ALLOW_REMOTE=1')
    base = Path(os.getenv('MEETING_AI_MEDIA_ROOT', str(Path.home() / 'MeetingMedia'))).expanduser().resolve()
    return AIConfig(
        enabled=_flag('MEETING_AI_ENABLED'), provider=provider, endpoint=endpoint,
        text_model=os.getenv('MEETING_AI_TEXT_MODEL', 'qwen3:8b'),
        embed_model=os.getenv('MEETING_AI_EMBED_MODEL', 'qwen3-embedding:0.6b'),
        vision_model=os.getenv('MEETING_AI_VISION_MODEL', ''),
        api_key=os.getenv('MEETING_AI_API_KEY', ''), media_root=base,
        max_upload_bytes=max(1024, min(int(os.getenv('MEETING_AI_MAX_UPLOAD_MB', '30')), 200) * 1024 * 1024),
        timeout_s=max(10, min(int(os.getenv('MEETING_AI_TIMEOUT_S', '360')), 1800)),
        enable_embeddings=_flag('MEETING_AI_EMBEDDINGS'),
        enable_vision=_flag('MEETING_AI_VISION'),
        enable_audio_import=_flag('MEETING_AI_AUDIO_IMPORT'),
    )

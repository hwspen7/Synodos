"""Validated infrastructure settings. Never print DATABASE_URL or other secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _integer(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = env.get(key, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def _boolean(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    value = env.get(key, str(default)).strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"{key} must be a boolean")
    return value in ("true", "1", "yes")


@dataclass(frozen=True)
class InfrastructureSettings:
    database_backend: str
    database_url: str
    database_pool_max: int
    inference_queue_size: int
    inference_timeout_s: int
    max_audio_buffer_seconds: int
    max_websocket_connections: int
    origins: tuple[str, ...]
    allow_nonbrowser_websocket: bool
    log_format: str
    log_level: str
    metrics_enabled: bool

    def safe_summary(self) -> dict:
        """Expose non-secret operational settings only."""
        return {
            "database_backend": self.database_backend,
            "database_pool_max": self.database_pool_max,
            "inference_queue_size": self.inference_queue_size,
            "max_audio_buffer_seconds": self.max_audio_buffer_seconds,
            "max_websocket_connections": self.max_websocket_connections,
            "metrics_enabled": self.metrics_enabled,
        }


def load_infrastructure_settings(env: Mapping[str, str] | None = None) -> InfrastructureSettings:
    e = os.environ if env is None else env
    backend = e.get("MEETING_STORAGE_BACKEND", "legacy").lower().strip()
    if backend not in ("legacy", "postgres"):
        raise ValueError("MEETING_STORAGE_BACKEND must be legacy or postgres")
    database_url = e.get("DATABASE_URL", "").strip()
    if backend == "postgres" and not database_url:
        raise ValueError("DATABASE_URL is required for PostgreSQL storage")
    if database_url and not database_url.startswith(("postgresql://", "postgres://")):
        raise ValueError("DATABASE_URL must use postgresql:// or postgres://")
    origins = tuple(x.strip().rstrip("/") for x in e.get(
        "APP_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if x.strip())
    if not origins or any(not v.startswith(("http://", "https://")) or "*" in v for v in origins):
        raise ValueError("APP_ALLOWED_ORIGINS must list explicit HTTP(S) origins")
    fmt = e.get("APP_LOG_FORMAT", "json").strip().lower()
    if fmt not in ("json", "plain"):
        raise ValueError("APP_LOG_FORMAT must be json or plain")
    level = e.get("APP_LOG_LEVEL", "INFO").strip().upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Invalid APP_LOG_LEVEL")
    return InfrastructureSettings(
        database_backend=backend,
        database_url=database_url,
        database_pool_max=_integer(e, "DB_POOL_MAX", 5, 1, 40),
        inference_queue_size=_integer(e, "INFERENCE_QUEUE_SIZE", 32, 1, 1024),
        inference_timeout_s=_integer(e, "INFERENCE_TIMEOUT_S", 180, 1, 3600),
        max_audio_buffer_seconds=_integer(e, "MAX_AUDIO_BUFFER_SECONDS", 120, 10, 3600),
        max_websocket_connections=_integer(e, "MAX_WS_CONNECTIONS", 12, 1, 10000),
        origins=origins,
        allow_nonbrowser_websocket=_boolean(e, "APP_ALLOW_NO_ORIGIN_WS"),
        log_format=fmt,
        log_level=level,
        metrics_enabled=_boolean(e, "APP_ENABLE_METRICS"),
    )

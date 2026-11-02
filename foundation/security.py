"""Local-first HTTP and WebSocket boundary checks."""

from __future__ import annotations

from urllib.parse import urlsplit

from foundation.settings import InfrastructureSettings


def origin_allowed(origin: str | None, settings: InfrastructureSettings) -> bool:
    if origin is None:
        return settings.allow_nonbrowser_websocket
    try:
        p = urlsplit(origin)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        if p.username or p.password or p.path or p.query or p.fragment:
            return False
        clean = f"{p.scheme}://{p.netloc}".rstrip("/")
        return clean in settings.origins
    except ValueError:
        return False


def public_error(exc: Exception) -> str:
    """Do not send database URLs or internal tracebacks to browsers."""
    return "An internal service error occurred"

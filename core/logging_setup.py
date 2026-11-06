"""Privacy-conscious JSON log formatting and trace IDs."""

from __future__ import annotations

import datetime
import json
import logging
import re
import traceback
import uuid


_SECRET = re.compile(r"(?i)(postgres(?:ql)?://)[^\s]+")


def sanitize(message: str) -> str:
    return _SECRET.sub(r"\1[REDACTED]", message)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        value = {
            "time": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize(record.getMessage()),
        }
        if hasattr(record, "request_id"):
            value["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            value["session_id"] = record.session_id
        if record.exc_info:
            value["exception_type"] = record.exc_info[0].__name__
            value["exception_message"] = sanitize(str(record.exc_info[1]))[:2000]
            value["exception_traceback"] = sanitize("".join(
                traceback.format_exception(*record.exc_info)
            ))[-7000:]
        return json.dumps(value, ensure_ascii=True)


def configure_logging(level: str, fmt: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    root.handlers.clear()
    root.addHandler(handler)


def request_id(value: str | None) -> str:
    if value and len(value) <= 64 and all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in value):
        return value
    return uuid.uuid4().hex

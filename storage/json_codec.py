"""JSON-safe representation of model state without writing new pickle blobs."""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numbers cannot be persisted")
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    raise TypeError(f"Unsupported persisted value type: {type(value).__name__}")

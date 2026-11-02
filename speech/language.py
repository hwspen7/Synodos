"""Normalize provider language names to ISO-style short codes where possible."""
from __future__ import annotations

NAMES = {
    "chinese": "zh", "mandarin": "zh", "english": "en", "cantonese": "yue",
    "french": "fr", "german": "de", "japanese": "ja", "korean": "ko",
    "spanish": "es", "russian": "ru", "italian": "it", "portuguese": "pt",
    "arabic": "ar", "hindi": "hi", "vietnamese": "vi", "thai": "th",
    "dutch": "nl", "turkish": "tr", "indonesian": "id", "malay": "ms",
}


def normalize_language(code: str | None) -> str:
    value = (code or "").strip().lower().replace("_", "-")
    return NAMES.get(value, value or "und")

"""Normalize provider language names to ISO-style short codes where possible."""
from __future__ import annotations

NAMES = {
    "chinese": "zh", "mandarin": "zh", "english": "en", "cantonese": "yue",
    "french": "fr", "german": "de", "japanese": "ja", "korean": "ko",
    "spanish": "es", "russian": "ru", "italian": "it", "portuguese": "pt",
    "arabic": "ar", "hindi": "hi", "bengali": "bn", "bangla": "bn",
    "vietnamese": "vi", "thai": "th",
    "dutch": "nl", "turkish": "tr", "indonesian": "id", "malay": "ms",
    "swedish": "sv", "danish": "da", "finnish": "fi",
    "polish": "pl", "czech": "cs", "filipino": "fil",
    "persian": "fa", "greek": "el", "hungarian": "hu",
    "macedonian": "mk", "romanian": "ro",
}


def normalize_language(code: str | None) -> str:
    value = (code or "").strip().lower().replace("_", "-")
    return NAMES.get(value, value or "und")


SUPPORTED_LANGUAGES = frozenset({
    "en", "zh", "hi", "es", "fr", "ar",
    "bn", "pt", "ru", "id", "de", "ja",
})


def normalize_supported_language(
    code: str | None, *, allow_auto: bool = False,
) -> str | None:
    """Normalize a language code and accept only the product's 12 languages."""
    value = normalize_language(code)
    if allow_auto and value == "auto":
        return "auto"
    primary = value.split("-", 1)[0]
    return primary if primary in SUPPORTED_LANGUAGES else None

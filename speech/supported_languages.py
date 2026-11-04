"""Phase 1 product language policy, independent of model capabilities.

Do not confuse languages decoded by Whisper/Qwen with languages for which
the product has reference-backed acceptance evidence. Keep provider aliases
for reading legacy transcripts, but never expose them as active support.
"""
from __future__ import annotations

from speech.language import normalize_language

PHASE1_LANGUAGE_CODES: tuple[str, ...] = (
    "en", "zh", "fr", "es", "ar", "id", "ja", "ru", "de", "hi",
)
PHASE1_LANGUAGE_SET = frozenset(PHASE1_LANGUAGE_CODES)


def spoken_language(code: str) -> str:
    """Normalize and validate requested audio source language, including auto."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Spoken language must be auto or a Phase 1 language code")
    selected = normalize_language(code)
    if selected in ("auto", *PHASE1_LANGUAGE_CODES):
        return selected
    raise ValueError(
        f"Spoken language {code!r} is not enabled in Phase 1. "
        "Choose one of en, zh, fr, es, ar, id, ja, ru, de, hi, or auto."
    )


def translation_target(code: str | None) -> str:
    """Empty means translation disabled; otherwise use supported target only."""
    if code is None or code == "":
        return ""
    if not isinstance(code, str):
        raise ValueError("Translation target must be a language code")
    selected = normalize_language(code)
    if selected in PHASE1_LANGUAGE_SET:
        return selected
    raise ValueError(
        f"Translation target {code!r} is not enabled in Phase 1. "
        "Choose one of en, zh, fr, es, ar, id, ja, ru, de, hi, or disable translation."
    )


def is_phase1_language(code: str | None) -> bool:
    return isinstance(code, str) and normalize_language(code) in PHASE1_LANGUAGE_SET

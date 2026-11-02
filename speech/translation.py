"""Offline translation via installed Argos language pair models."""
from __future__ import annotations


class TranslationUnavailable(RuntimeError):
    pass


def translate(text: str, source: str, target: str) -> str:
    if not text.strip() or source == target:
        return text
    if source in ("", "auto", "und"):
        raise TranslationUnavailable("Detected source language is required for translation")
    try:
        import argostranslate.translate as argos
    except ImportError as exc:
        raise TranslationUnavailable("Install argostranslate and an offline language-pair package") from exc
    installed = {lang.code: lang for lang in argos.get_installed_languages()}
    if source not in installed or target not in installed:
        raise TranslationUnavailable(f"Language model not installed: {source} -> {target}")
    pair = installed[source].get_translation(installed[target])
    if pair is None:
        raise TranslationUnavailable(f"Language pair unavailable: {source} -> {target}")
    return pair.translate(text)

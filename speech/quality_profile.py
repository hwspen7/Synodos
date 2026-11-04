"""Opt-in, reversible ASR coverage candidate for paired real-audio evaluation.

This is not an accuracy claim. It changes how verified voice is retained,
decoded, and retried, so every promotion requires frozen-reference scoring.
"""
from __future__ import annotations

import os


def quality_candidate_enabled() -> bool:
    """Read the process-wide quality candidate once per invocation.

    The default remains the frozen reference path. Explicit opt-in jointly
    enables loss-resistant word handling, gap retries, and buffered voice
    retention in the corresponding processing layers.
    """
    return os.getenv("MEETING_ASR_QUALITY_CANDIDATE", "0").strip().lower() in (
        "1", "true", "yes"
    )

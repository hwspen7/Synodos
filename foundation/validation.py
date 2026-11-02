"""Validate end-to-end audio and model operating thresholds."""

from __future__ import annotations

import math


def validate_application_settings(settings) -> None:
    if settings.sample_rate != 16000:
        raise ValueError("Only 16000 Hz audio is supported")
    bounds = {
        "vad_threshold_s": (0.5, 30.0),
        "rec_threshold_s": (0.2, 30.0),
        "merge_gap_s": (0.0, 120.0),
        "final_discard_s": (0.0, 120.0),
        "cluster_merge_thr": (0.0, 1.0),
        "min_speaker_audio_s": (0.2, 60.0),
        "spk_match_threshold": (0.0, 1.0),
    }
    for name, (minimum, maximum) in bounds.items():
        number = float(getattr(settings, name))
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

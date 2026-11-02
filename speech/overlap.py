"""Explicitly opt-in overlap detection; never fabricate separated speaker words."""
from __future__ import annotations
import numpy as np


class OverlapUnavailable(RuntimeError):
    pass


class OverlapDetector:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def detect(self, audio: np.ndarray) -> list[dict]:
        if self.pipeline is None:
            raise OverlapUnavailable("Overlap detection requires a configured pyannote pipeline")
        import torch
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        annotation = self.pipeline({"waveform": waveform, "sample_rate": 16000})
        return [{"start_ms": round(seg.start * 1000), "end_ms": round(seg.end * 1000),
                 "type": "overlapping_speech"} for seg in annotation.get_timeline().support()]

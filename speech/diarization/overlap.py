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
        output = self.pipeline({"waveform": waveform, "sample_rate": 16000})
        # Modern pyannote returns a DiarizeOutput wrapper. For accurate
        # overlaps use its full diarization, NOT the exclusive timeline.
        annotation = getattr(output, "speaker_diarization", output)
        if not callable(getattr(annotation, "get_overlap", None)):
            raise OverlapUnavailable(
                "Speaker annotation cannot report simultaneous speech"
            )
        intervals = annotation.get_overlap().support()
        result = []
        for seg in intervals:
            a, b = round(float(seg.start) * 1000), round(float(seg.end) * 1000)
            if 0 <= a < b <= round(len(audio) * 1000 / 16000) + 25:
                result.append({
                    "start_ms": a, "end_ms": b,
                    "type": "overlapping_speech",
                })
        return result

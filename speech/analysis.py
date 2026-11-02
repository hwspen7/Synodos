"""Optional acoustic analysis orchestrator; never emits invented events."""
from __future__ import annotations
import os
import threading
import numpy as np
from speech.config import SpeechSettings
from speech.enhancement import quality

_LOCK = threading.RLock()
_EVENTS = None
_OVERLAP = None


def analyze(samples: np.ndarray, settings: SpeechSettings) -> dict:
    global _EVENTS, _OVERLAP
    result = {"quality": quality(samples), "overlap": [], "events": [], "emotion": None,
              "separated_candidates": []}
    with _LOCK:
        if settings.enable_events:
            if _EVENTS is None:
                from speech.events import SenseVoiceEvents
                _EVENTS = SenseVoiceEvents(os.getenv("MEETING_SENSEVOICE_MODEL", "iic/SenseVoiceSmall"))
            event_data = _EVENTS.analyze(samples)
            result["events"] = event_data["events"]
            result["emotion"] = event_data["emotion"]
        if settings.enable_overlap:
            if _OVERLAP is None:
                from speech.overlap import OverlapDetector
                model = os.getenv("MEETING_OVERLAP_MODEL")
                if not model:
                    raise RuntimeError("MEETING_OVERLAP_MODEL must reference an installed pyannote pipeline")
                from pyannote.audio import Pipeline
                _OVERLAP = OverlapDetector(Pipeline.from_pretrained(model))
            result["overlap"] = _OVERLAP.detect(samples)
            if result["overlap"] and os.getenv("MEETING_SEPARATE_OVERLAP", "false").lower() in ("1", "true", "yes"):
                from speech.separation import SepformerSeparator
                from speech.engine import recognize
                separator = SepformerSeparator(os.getenv("MEETING_SEPARATION_MODEL", "speechbrain/sepformer-wsj02mix"))
                for event in result["overlap"][:4]:
                    start = max(0, int(event["start_ms"] * 16))
                    end = min(len(samples), int(event["end_ms"] * 16))
                    if end - start < 8000 or end - start > 480000:
                        continue
                    parts = separator.separate(samples[start:end])
                    for i, component in enumerate(parts):
                        guess = recognize(component, config=settings, language=settings.language)
                        result["separated_candidates"].append({
                            "source_index": i + 1,
                            "start_ms": event["start_ms"],
                            "end_ms": event["end_ms"],
                            "text": " ".join(piece.text for piece in guess.utterances),
                            "verified": False,
                        })
    return result

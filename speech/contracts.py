"""Provider-neutral transcription contracts with timestamp provenance."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Utterance:
    text: str
    start_ms: int
    end_ms: int
    language: str = "und"
    words: tuple[Word, ...] = ()
    timestamp_quality: str = "segment"
    confidence: float | None = None
    language_confidence: float | None = None
    recognition_origin: str = "primary"

    def as_legacy(self) -> dict:
        return {
            "start": self.start_ms,
            "end": self.end_ms,
            "text": self.text,
            "language": self.language,
            "timestamp_quality": self.timestamp_quality,
            "words": [vars(x) for x in self.words],
            "confidence": self.confidence,
            "language_confidence": self.language_confidence,
            "recognition_origin": self.recognition_origin,
        }


@dataclass(frozen=True)
class Recognition:
    utterances: tuple[Utterance, ...] = field(default_factory=tuple)
    language: str = "und"
    confidence: float | None = None
    engine: str = "legacy"
    speech_detected: bool | None = None

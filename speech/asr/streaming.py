"""Two distinct streaming implementations: speculative window previews and native ONNX."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Preview:
    text: str
    language: str
    start_ms: int
    end_ms: int
    revision: int
    final: bool = False


class PreviewTracker:
    """Ephemeral transcript revisions never enter persisted confirmed segments."""
    def __init__(self):
        self.revision = 0
        self.last = ("", "")

    def update(self, text: str, language: str, offset_ms: int, end_ms: int) -> Preview | None:
        candidate = (text.strip(), language)
        if not candidate[0] or candidate == self.last:
            return None
        self.last = candidate
        self.revision += 1
        return Preview(candidate[0], language, offset_ms, end_ms, self.revision)

    def clear(self, offset_ms: int = 0, end_ms: int = 0) -> Preview | None:
        """Emit one monotonic removal event for a previously visible draft."""
        if not self.last[0]:
            return None
        self.last = ("", "")
        self.revision += 1
        return Preview("", "und", offset_ms, end_ms, self.revision)



class NativeOnlineRecognizer:
    """Requires installed sherpa-onnx and explicit streaming transducer weights."""
    def __init__(self, encoder: str, decoder: str, joiner: str, tokens: str):
        if not all((encoder, decoder, joiner, tokens)):
            raise ValueError("Native streaming requires encoder, decoder, joiner and tokens")
        import sherpa_onnx
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=encoder, decoder=decoder, joiner=joiner,
            sample_rate=16000, feature_dim=80,
        )
        self.stream = self.recognizer.create_stream()

    def accept(self, audio: np.ndarray) -> str:
        self.stream.accept_waveform(16000, np.asarray(audio, dtype=np.float32))
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return str(self.recognizer.get_result(self.stream).text)

    def finish(self) -> str:
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return str(self.recognizer.get_result(self.stream).text)

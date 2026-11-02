"""Optional sound event tagging; do not infer emotion without a real model."""
from __future__ import annotations
import numpy as np


class SenseVoiceEvents:
    def __init__(self, model_path: str = "iic/SenseVoiceSmall"):
        self.model_path = model_path
        self.model = None

    def analyze(self, audio: np.ndarray) -> dict:
        if len(audio) < 3200:
            return {"emotion": None, "events": []}
        if self.model is None:
            from funasr import AutoModel
            self.model = AutoModel(model=self.model_path, disable_update=True, disable_log=True)
        results = self.model.generate(input=np.asarray(audio, dtype=np.float32), disable_pbar=True, language="auto", use_itn=True)
        raw = str(results[0].get("text", "") if results else "")
        import re
        markers = re.findall(r"<\|([^|]+)\|>", raw)
        emotion = next((v for v in markers if v in {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}), None)
        return {"emotion": emotion, "events": [v for v in markers if v in {"Speech", "Laughter", "Applause", "Music", "Crying"}]}

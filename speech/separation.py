"""Experimental two-speaker source separation for flagged overlap windows.

Outputs are *candidate streams*, never automatically inserted as verified transcript.
"""
from __future__ import annotations
import threading
import numpy as np


class SepformerSeparator:
    def __init__(self, model_id: str = "speechbrain/sepformer-wsj02mix"):
        self.model_id = model_id
        self._model = None
        self._lock = threading.RLock()

    def separate(self, audio16k: np.ndarray) -> list[np.ndarray]:
        from scipy.signal import resample_poly
        import torch
        audio16k = np.asarray(audio16k, dtype=np.float32)
        if len(audio16k) < 8000 or len(audio16k) > 480000:
            raise ValueError("Overlap window must contain between 0.5 and 30 seconds of audio")
        with self._lock:
            if self._model is None:
                from speechbrain.inference.separation import SepformerSeparation
                self._model = SepformerSeparation.from_hparams(source=self.model_id)
            # The public WSJ0-2mix checkpoint expects 8 kHz mono audio.
            input8k = resample_poly(audio16k, 1, 2).astype(np.float32)
            sources = self._model.separate_batch(torch.from_numpy(input8k)[None, :])
        tensor = sources.detach().cpu().numpy()
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 2:
            raise RuntimeError("Unexpected speech separation output shape")
        return [resample_poly(tensor[0, :, i], 2, 1).astype(np.float32) for i in range(2)]

"""Offline verification of Paraformer, VAD, punctuation, and CAM++ diarization."""

import sys
import time
from pathlib import Path

from funasr import AutoModel

# Reuse the backend model alias and local cache resolver.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import resolve_model  # noqa: E402

WAV = Path(__file__).resolve().parent / "Genshin-1222a.wav"

model = AutoModel(
    model=resolve_model("paraformer-zh"),
    vad_model=resolve_model("fsmn-vad"),  # Internal VAD
    punc_model=resolve_model("ct-punc-c"),
    spk_model=resolve_model("cam++"),  # Speaker embeddings and clustering
    disable_log=True,
    disable_update=True,
)

t0 = time.time()
res = model.generate(
    input=str(WAV),
    batch_size_s=300,
    sentence_timestamp=True,
)

for sent in res[0]["sentence_info"]:
    # Start/end timestamps are milliseconds.
    print(
        f"[Speaker {sent['spk']}] "
        f"({sent['start']/1000:.2f}s-{sent['end']/1000:.2f}s): {sent['text']}"
    )

t1 = time.time()
print("Elapsed time:", t1 - t0, "s")

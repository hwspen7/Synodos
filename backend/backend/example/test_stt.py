"""离线验证模型可用性：paraformer + fsmn-vad + 标点 + cam++ 说话人分段。"""

import sys
import time
from pathlib import Path

from funasr import AutoModel

# 复用后端配置里的模型名 -> 本地缓存路径解析，避免 funasr 短名注册问题
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.config import resolve_model  # noqa: E402

WAV = Path(__file__).resolve().parent / "Genshin-1222a.wav"

model = AutoModel(
    model=resolve_model("paraformer-zh"),
    vad_model=resolve_model("fsmn-vad"),   # 原生支持，内部切段
    punc_model=resolve_model("ct-punc-c"),
    spk_model=resolve_model("cam++"),       # 内部提 embedding + 聚类
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
    # start/end 单位是毫秒
    print(
        f"[Speaker {sent['spk']}] "
        f"({sent['start']/1000:.2f}s-{sent['end']/1000:.2f}s): {sent['text']}"
    )

t1 = time.time()
print("Elapsed time:", t1 - t0, "s")

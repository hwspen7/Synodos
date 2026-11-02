"""通过 WebSocket 向后端实时推送示例音频，验证 VAD -> 聚类 -> ASR 全流程。

后端只推送可信段列表；脚本按“同说话人 + 相邻间隔 < merge_gap_s”本地合并为段落
（与前端段落视图一致）后打印。

用法：
  1. 从仓库根目录启动后端：uv run uvicorn main:app --port 8000
  2. python example/stream_test.py
"""

import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


def build_paragraphs(segments: list[dict], merge_gap_s: float = 2.0) -> list[dict]:
    """由可信段生成段落：同一说话人 + 间隔小于 merge_gap_s 才合并（与前端一致）。"""
    gap_ms = int(merge_gap_s * 1000)
    paras: list[dict] = []
    for seg in sorted(segments, key=lambda s: s["start_ms"]):
        last = paras[-1] if paras else None
        if (
            last is not None
            and seg["speaker_id"] == last["speaker_id"]
            and seg["start_ms"] - last["end_ms"] < gap_ms
        ):
            last["end_ms"] = seg["end_ms"]
            last["text"] += seg["text"]
        else:
            paras.append(
                {
                    "speaker_id": seg["speaker_id"],
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "text": seg["text"],
                }
            )
    return paras


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/audio")
    ap.add_argument("--wav", default="example/Genshin-1222a.wav")
    ap.add_argument("--segment-s", type=float, default=4.0)
    ap.add_argument("--silence-s", type=float, default=3.0)
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype("float32")

    # 拼接：语音段 + 静音 两次，模拟“说一句话、停顿一下”
    speech = audio[: int(args.segment_s * 16000)]
    silence = np.zeros(int(args.silence_s * 16000), dtype=np.float32)
    stream = np.concatenate([speech, silence, speech, silence])
    pcm = (stream * 32767).astype(np.int16)
    print(f"stream {len(pcm) / 16000:.1f}s, sending chunks...")

    async with websockets.connect(args.url) as ws:
        async def reader() -> None:
            merge_gap_s = 2.0
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "ready":
                    merge_gap_s = float(msg.get("merge_gap_s") or merge_gap_s)
                    print("ready:", msg)
                elif kind == "transcript":
                    segments = msg["segments"]
                    paragraphs = build_paragraphs(segments, merge_gap_s)
                    print(
                        f"[transcript v{msg['version']}] {len(segments)} 片段 / "
                        f"{len(paragraphs)} 段"
                    )
                    for p in paragraphs:
                        text = p["text"] or "（无内容）"
                        print(
                            f"  [说话人{p['speaker_id']}] "
                            f"{p['start_ms']}ms-{p['end_ms']}ms: {text}"
                        )
                elif kind == "state":
                    print(f"[state] buffered {msg['buffered_ms']}ms")
                elif kind == "closed":
                    print("server closed")
                    return

        task = asyncio.create_task(reader())
        chunk = 1600  # 100ms
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i : i + chunk].tobytes())
            await asyncio.sleep(0.02)
        # 请求收尾：后端处理剩余音频后推送最终 transcript 并关闭
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(task, timeout=60)
        except asyncio.TimeoutError:
            print("timeout waiting for server to close")
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())

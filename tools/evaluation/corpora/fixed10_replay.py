"""Replay every frozen Fixed10 WAV through the full meeting WebSocket pipeline.

Uses ordered MOPA v1 frames and wall-clock pacing. Fails closed on missing
acknowledgements, server errors, incomplete finalization or changed WAV bytes.
The per-speaker output for clips 09/10 feeds fixed10_dashboard.py directly.

Requires a running backend, websockets; no offline dummy inference is allowed.
"""
from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import struct
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import wave

_FRAME = struct.Struct(">4sIQ")
_CHUNK_SAMPLES = 1600


def _frame(seq: int, offset: int, pcm: bytes) -> bytes:
    if len(pcm) % 2:
        raise ValueError("Truncated PCM16 frame")
    return _FRAME.pack(b"MOPA", seq, offset) + pcm


def _pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError(f"Input must be 16kHz mono 16bit WAV: {path}")
        return audio.readframes(audio.getnframes())


def _url(base: str) -> str:
    u = urlsplit(base)
    if u.scheme not in ("ws", "wss") or not u.netloc or not u.path.endswith("/ws/audio"):
        raise ValueError("Use a ws:// or wss:// URL ending in /ws/audio")
    params = dict(parse_qsl(u.query, keep_blank_values=True))
    params.update({
        "language": "auto",
        "translate_to": "",
        "record_audio": "false",
        "vad_threshold": "4",
        "rec_threshold": "1.5",
    })
    params.pop("meeting_id", None)
    params.pop("session_id", None)
    return urlunsplit((u.scheme, u.netloc, u.path, urlencode(params), u.fragment))


async def replay_one(case: dict, root: Path, url: str, *, pace: float = 1.0) -> dict:
    import websockets
    wav = root / case["audio"]
    if not wav.is_file():
        raise FileNotFoundError(wav)
    if sha256(wav.read_bytes()).hexdigest() != case["sha256"]:
        raise ValueError(f"Changed benchmark WAV: {case['id']}")
    pcm = _pcm(wav)
    sent = len(pcm) // 2
    if not sent:
        raise ValueError("Cannot replay an empty WAV")
    snapshot: list[dict] = []
    events = []
    errors = []
    acknowledged = None
    final_received = False
    closed = False
    ready = asyncio.Event()
    started = time.perf_counter()
    async with websockets.connect(url, max_size=16*1024*1024, ping_interval=20) as ws:
        async def messages():
            nonlocal snapshot, acknowledged, final_received, closed
            async for raw in ws:
                if not isinstance(raw, str):
                    errors.append("Unexpected binary server message")
                    break
                msg = json.loads(raw)
                typ = msg.get("type", "unknown")
                events.append(typ)
                if typ == "ready":
                    ready.set()
                elif typ == "transcript":
                    snapshot = msg.get("segments", [])
                elif typ == "audio_received":
                    acknowledged = msg.get("samples")
                    final_received = bool(msg.get("matched"))
                elif typ == "error":
                    errors.append(str(msg.get("message", "server error")))
                elif typ == "closed":
                    closed = True
                    return
        listener = asyncio.create_task(messages())
        try:
            await asyncio.wait_for(ready.wait(), timeout=45)
            period = _CHUNK_SAMPLES / 16000 / pace
            begun = time.monotonic()
            for seq, start in enumerate(range(0, sent, _CHUNK_SAMPLES)):
                a, b = start * 2, min(sent, start + _CHUNK_SAMPLES) * 2
                await ws.send(_frame(seq, start, pcm[a:b]))
                due = begun + (seq + 1) * period
                await asyncio.sleep(max(0, due - time.monotonic()))
            await ws.send(json.dumps({"type": "stop", "expected_samples": sent}))
            await asyncio.wait_for(listener, timeout=180)
        finally:
            if not listener.done():
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
    elapsed_ms = round((time.perf_counter()-started)*1000, 1)
    if errors or not final_received or acknowledged != sent or not closed:
        raise RuntimeError(
            f"{case['id']} failed end-to-end integrity: errors={errors}, "
            f"acknowledged={acknowledged}, sent={sent}, matched={final_received}, closed={closed}"
        )
    return {
        "id": case["id"], "kind": case["kind"],
        "language": case["language"], "duration_s": case["duration_s"],
        "sent_samples": sent, "acknowledged_samples": acknowledged,
        "wall_time_ms": elapsed_ms, "segments": snapshot,
        "status": "completed", "events_count": len(events),
    }


async def run(root: Path, url: str, pace: float) -> int:
    lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
    cases = lock["cases"]
    if len(cases) != 10:
        raise ValueError("Expected 10 immutable cases")
    actual_url = _url(url)
    outputs: list[dict] = []
    overlaps: dict[str, dict] = {}
    for case in cases:
        print(f"Replaying {case['id']} ({case['kind']}, {case['language']})", flush=True)
        try:
            result = await replay_one(case, root, actual_url, pace=pace)
            outputs.append(result)
            if case["kind"] == "overlap":
                overlaps[case["id"]] = {"segments": result["segments"]}
        except Exception as exc:
            outputs.append({
                "id": case["id"], "kind": case["kind"],
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
            if case["kind"] == "overlap":
                overlaps[case["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
    (root / "replay_results.json").write_text(json.dumps({
        "schema": "meeting-fixed10-replay-v1", "cases": outputs,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "overlap_predictions.json").write_text(
        json.dumps(overlaps, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    failures = [x for x in outputs if x.get("status") != "completed"]
    print(f"Replay complete: {len(outputs)-len(failures)}/{len(outputs)} successful")
    for failure in failures:
        print(f"{failure['id']} failed: {failure['error']}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed10"))
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/audio")
    parser.add_argument("--pace", type=float, default=1.0,
                        help="1.0 = wall-clock realtime playback")
    args = parser.parse_args()
    if not 0.25 <= args.pace <= 2.0:
        parser.error("Playback pace must be between 0.25 and 2.0")
    return asyncio.run(run(args.root, args.url, args.pace))


if __name__ == "__main__":
    raise SystemExit(main())

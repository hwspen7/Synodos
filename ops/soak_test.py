"""Bounded live WebSocket streaming smoke/soak test against a test database.

This exercises real HTTP/WebSocket paths and model dispatch with synthetic PCM.
It does not measure ASR accuracy or guarantee a production load capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class SoakError(RuntimeError):
    pass


def guard_target(http_url: str, websocket_url: str) -> None:
    http = urlparse(http_url)
    ws = urlparse(websocket_url)
    for target in (http, ws):
        if target.hostname not in ("localhost", "127.0.0.1"):
            raise SoakError("Remote hosts are not allowed by the local test harness")
        if target.port != 8001:
            raise SoakError("The load test is restricted to port 8001, not the live port 8000")
    if http.scheme != "http" or ws.scheme != "ws" or ws.path != "/ws/audio":
        raise SoakError("Unexpected local HTTP or WebSocket endpoint")


def api(url: str, method: str = "GET", body: dict | None = None) -> dict:
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=raw, method=method,
                  headers={"Content-Type": "application/json"} if raw is not None else {})
    with urlopen(req, timeout=20) as response:
        result = response.read()
    return json.loads(result) if result else {}


def pcm_frame(sample_rate: int, frame_ms: int, phase: int) -> bytes:
    """Deterministic quiet sine wave, not actual spoken language."""
    n = sample_rate * frame_ms // 1000
    # A low amplitude, continuous signal avoids high-volume playback.
    return struct.pack("<" + "h" * n, *(
        round(250 * math.sin(2 * math.pi * 220 * (phase + i) / sample_rate))
        for i in range(n)
    ))


async def run_session(base_url: str, ws_url: str, seconds: float, frame_ms: int,
                      origin: str, index: int, interval: float) -> dict:
    from websockets.asyncio.client import connect

    meeting = await asyncio.to_thread(
        api, f"{base_url}/api/meetings", "POST", {"title": f"Soak test {index}"}
    )
    mid = meeting["meeting"]["id"]
    received: list[str] = []
    errors: list[str] = []
    started = time.monotonic()
    try:
        address = f"{ws_url}?" + urlencode({"meeting_id": mid, "vad_threshold": "10", "rec_threshold": "2"})
        async with connect(address, origin=origin, open_timeout=30, close_timeout=10,
                           max_size=5_000_000, ping_interval=20) as ws:
            initial = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if initial.get("type") != "ready":
                raise SoakError(f"Session {index} did not receive the ready handshake")
            received.append("ready")
            count = math.ceil(seconds * 1000 / frame_ms)
            samples_per_frame = 16000 * frame_ms // 1000
            for i in range(count):
                await ws.send(pcm_frame(16000, frame_ms, i * samples_per_frame))
                if interval:
                    await asyncio.sleep(frame_ms / 1000 * interval)
            await ws.send(json.dumps({"type": "stop"}))
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=min(20, deadline - time.monotonic()))
                except (asyncio.TimeoutError, Exception) as exc:
                    # Normal clean close after a "closed" notification is expected.
                    if "closed" in received:
                        break
                    raise SoakError(f"Session {index} disconnected before completion: {type(exc).__name__}") from exc
                if isinstance(data, str):
                    msg = json.loads(data)
                    kind = msg.get("type", "unknown")
                    received.append(kind)
                    if kind == "error" or (kind == "closed" and msg.get("saved") is False):
                        errors.append(str(msg.get("message") or "backend error"))
                    if kind == "closed":
                        break
            if "closed" not in received:
                raise SoakError(f"Session {index} timed out waiting for final processing")
            if errors:
                raise SoakError(f"Session {index} reported backend errors: {errors[:3]}")
        detail = await asyncio.to_thread(api, f"{base_url}/api/meetings/{mid}")
        if detail.get("meeting", {}).get("id") != mid:
            raise SoakError(f"Session {index} meeting was not readable after saving")
        return {"session": index, "meeting_id": mid, "elapsed_seconds": round(time.monotonic() - started, 3),
                "messages": {kind: received.count(kind) for kind in set(received)}}
    finally:
        try:
            await asyncio.to_thread(api, f"{base_url}/api/meetings/{mid}", "DELETE")
        except Exception:
            pass


async def run(options: argparse.Namespace) -> list[dict]:
    guard_target(options.http_url, options.ws_url)
    if not options.confirm_test_database:
        raise SoakError("Use --confirm-test-database only after starting a separate server on meeting_test at port 8001")
    if not 1 <= options.clients <= 4 or not 1 <= options.sessions <= 100:
        raise SoakError("clients must be 1..4 and sessions 1..100")
    if not 1 <= options.seconds <= 600 or options.frame_ms not in (20, 40, 100):
        raise SoakError("seconds must be 1..600 and frame-ms must be 20, 40 or 100")
    health = await asyncio.to_thread(api, f"{options.http_url}/api/health")
    info = health.get("infrastructure") or {}
    if info.get("database_backend") != "postgres":
        raise SoakError("Test service did not report PostgreSQL as its backend")
    semaphore = asyncio.Semaphore(options.clients)

    async def one(i: int) -> dict:
        async with semaphore:
            return await run_session(options.http_url, options.ws_url, options.seconds,
                                     options.frame_ms, options.origin, i, options.pace)

    return await asyncio.gather(*(one(i) for i in range(options.sessions)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Live synthetic PCM stress test for isolated local backend")
    parser.add_argument("--http-url", default="http://127.0.0.1:8001")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8001/ws/audio")
    parser.add_argument("--origin", default="http://127.0.0.1:5173")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--confirm-test-database", action="store_true")
    options = parser.parse_args()
    try:
        result = asyncio.run(run(options))
        print(json.dumps({"result": "passed", "sessions": result}, indent=2))
    except (SoakError, OSError, ValueError, TimeoutError, KeyError) as exc:
        print(f"SOAK TEST FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

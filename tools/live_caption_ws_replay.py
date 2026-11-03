"""Replay a real local conversation at microphone rate over Meeting WebSocket.

Measure time to first provisional and confirmed subtitles, revision order,
audio acknowledgments, final segment count, speaker relabel churn, and stop
completion. This verifies the actual server WebSocket/caption pipeline.

This is an operational replay, NOT human reference-backed ASR/DER accuracy.
Loops repeat the SAME source audio for endurance testing and are not new
independent meetings. No text/audio is stored in the report by default.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fixed150_prepare import atomic_json
from tools.phase07_benchmark import load_wav

HEADER = struct.Struct(">4sIQ")
FRAME_SAMPLES = 3200
SAMPLE_RATE = 16000
SCHEMA = "meeting-live-caption-websocket-replay-v1"


def pcm_packet(pcm, seq: int, offset: int) -> bytes:
    import numpy as np

    samples = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if not len(samples) or len(samples) > 16000:
        raise ValueError("PCM frame must contain 1..16000 16 kHz samples")
    if seq < 0 or offset < 0:
        raise ValueError("Sequence and absolute sample offset must be nonnegative")
    return HEADER.pack(b"MOPA", seq, offset) + samples.astype("<i2").tobytes()


def capture_event(state: dict, message: dict, elapsed_ms: float) -> bool:
    """Count events without writing private transcript strings to disk."""
    kind = str(message.get("type", "unknown"))
    state["events"][kind] += 1
    if kind == "partial":
        revision = message.get("revision")
        if not isinstance(revision, int) or revision <= state["last_partial_revision"]:
            state["flags"].append("nonmonotonic_partial_revision")
        else:
            state["last_partial_revision"] = revision
        text = str(message.get("text", "") or "").strip()
        if text and state["first_preview_ms"] is None:
            state["first_preview_ms"] = round(elapsed_ms, 1)
        if not text:
            state["clear_events"] += 1
        state["last_preview_chars"] = len(text)
    elif kind == "transcript":
        version = message.get("version")
        if not isinstance(version, int) or version <= state["last_transcript_version"]:
            state["flags"].append("nonmonotonic_transcript_version")
        else:
            state["last_transcript_version"] = version
        incoming = message.get("segments") or []
        if incoming and state["first_confirmed_ms"] is None:
            state["first_confirmed_ms"] = round(elapsed_ms, 1)
        previous = state["segment_speakers"]
        updated = {}
        for segment in incoming:
            sid = segment.get("id")
            if not isinstance(sid, int):
                state["flags"].append("invalid_segment_id")
                continue
            voice = segment.get("speaker_id")
            updated[sid] = voice
            if sid in previous and previous[sid] != voice:
                state["speaker_reassignments"] += 1
        if len(updated) != len(incoming):
            state["flags"].append("duplicate_segment_ids")
        # No semantic inference: repeated words can be genuine repeated speech.
        state["segment_speakers"] = updated
        state["final_segment_count"] = len(updated)
    elif kind == "audio_received":
        state["audio_delivery_matched"] = message.get("matched") is True
        state["audio_received_samples"] = message.get("samples")
    elif kind == "closed":
        state["server_closed"] = True
        state["server_saved"] = message.get("saved")
        return True
    elif kind == "error":
        state["errors"].append(str(message.get("message", ""))[:500])
    return False


def new_capture() -> dict:
    return {
        "events": Counter(), "flags": [], "errors": [],
        "first_preview_ms": None, "first_confirmed_ms": None,
        "last_partial_revision": 0, "last_transcript_version": 0,
        "last_preview_chars": 0, "clear_events": 0,
        "segment_speakers": {}, "speaker_reassignments": 0,
        "final_segment_count": 0, "server_closed": False,
        "server_saved": None, "audio_delivery_matched": None,
        "audio_received_samples": None,
    }


async def replay(args) -> dict:
    import numpy as np
    from websockets.asyncio.client import connect

    source = Path(args.audio)
    if not source.is_file():
        raise FileNotFoundError(f"Missing test audio: {source}")
    wave = load_wav(source)
    if not len(wave):
        raise ValueError("Cannot replay empty audio")
    if args.loops < 1 or args.loops > 100:
        raise ValueError("--loops must be between 1 and 100")
    if not 0.25 <= args.speed <= 8:
        raise ValueError("--speed must be 0.25..8")
    if not 0 <= args.gap_ms <= 3000:
        raise ValueError("--gap-ms must be 0..3000")
    gap = np.zeros(round(args.gap_ms * SAMPLE_RATE / 1000), dtype=np.int16)
    chunks = [
        np.clip(np.round(wave * 32768), -32768, 32767).astype("<i2")
    ]

    counters = new_capture()
    sent_samples = 0
    sent_sequence = 0
    first_send = None
    ready = None
    started = time.perf_counter()
    async with connect(args.url, origin=args.origin, max_size=8 * 1024 * 1024) as ws:
        opening = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if opening.get("type") != "ready":
            raise RuntimeError(f"Server did not send ready: {opening.get('type')}")
        ready = {
            "engine": opening.get("asr_engine"),
            "model": opening.get("asr_model"),
            "processing_window_s": opening.get("processing_window_s"),
            "min_commit_context_s": opening.get("min_commit_context_s"),
        }

        async def receive():
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                elapsed = 0.0 if first_send is None else (
                    time.perf_counter() - first_send) * 1000
                if capture_event(counters, msg, elapsed):
                    return
            if not counters["server_closed"]:
                counters["flags"].append("socket_closed_before_final_confirmation")

        receiver = asyncio.create_task(receive())
        try:
            for loop_index in range(args.loops):
                for audio in chunks + ([gap] if loop_index + 1 < args.loops
                                       and len(gap) else []):
                    for offset in range(0, len(audio), FRAME_SAMPLES):
                        pcm = audio[offset:offset + FRAME_SAMPLES]
                        if first_send is None:
                            first_send = time.perf_counter()
                        packet = pcm_packet(pcm, sent_sequence, sent_samples)
                        await ws.send(packet)
                        sent_sequence += 1
                        sent_samples += len(pcm)
                        await asyncio.sleep(len(pcm) / SAMPLE_RATE / args.speed)
            await ws.send(json.dumps({
                "type": "stop", "expected_samples": sent_samples
            }))
            await asyncio.wait_for(receiver, timeout=args.final_timeout)
        finally:
            if not receiver.done():
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    summary = {
        "schema": SCHEMA,
        "source_audio": str(source),
        "source_audio_duration_s": round(len(wave) / SAMPLE_RATE, 2),
        "loops": args.loops, "gap_ms": args.gap_ms, "playback_speed": args.speed,
        "real_time_latency_evaluable": args.speed == 1.0,
        "sent_frames": sent_sequence, "sent_samples": sent_samples,
        "sent_audio_duration_s": round(sent_samples / SAMPLE_RATE, 2),
        "total_wall_ms": elapsed_ms,
        "server_ready": ready,
        "events": dict(counters["events"]),
        "first_preview_wall_ms": counters["first_preview_ms"],
        "first_confirmed_wall_ms": counters["first_confirmed_ms"],
        "final_confirmed_segments": counters["final_segment_count"],
        "speaker_reassignments": counters["speaker_reassignments"],
        "preview_clear_events": counters["clear_events"],
        "audio_received_samples": counters["audio_received_samples"],
        "delivery_matched": counters["audio_delivery_matched"],
        "server_closed": counters["server_closed"],
        "server_saved": counters["server_saved"],
        "flags": counters["flags"],
        "errors": counters["errors"],
        "voiceprint_accuracy": None,
        "asr_cer": None,
        "subtitle_completeness": "manual_review_required",
    }
    if counters["audio_received_samples"] not in (None, sent_samples):
        summary["flags"].append("server_audio_sample_count_mismatch")
    if not counters["server_closed"]:
        summary["flags"].append("server_did_not_finish")
    if args.output:
        atomic_json(args.output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path,
                        default=Path("benchmark/fixed195/audio/fixed195-en-m01.wav"))
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/audio?"
                        "language=auto&record_audio=false")
    parser.add_argument("--origin", default="http://localhost:5173")
    parser.add_argument("--loops", type=int, default=1,
                        help="Repeat same source for endurance only, not accuracy")
    parser.add_argument("--gap-ms", type=int, default=500)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Only speed=1.0 gives real-time timing")
    parser.add_argument("--final-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/fixed195/live_caption_replay.json"))
    args = parser.parse_args()
    summary = asyncio.run(replay(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if (summary["server_closed"] and summary["delivery_matched"]
                 and not summary["errors"] and
                 not summary["flags"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

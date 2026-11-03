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
import math
from statistics import median
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
DEFAULT_WS_URL = (
    "ws://127.0.0.1:8000/ws/audio?"
    "language=auto&record_audio=false&translate_to="
)


def pcm_packet(pcm, seq: int, offset: int) -> bytes:
    import numpy as np

    samples = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if not len(samples) or len(samples) > 16000:
        raise ValueError("PCM frame must contain 1..16000 16 kHz samples")
    if seq < 0 or offset < 0:
        raise ValueError("Sequence and absolute sample offset must be nonnegative")
    return HEADER.pack(b"MOPA", seq, offset) + samples.astype("<i2").tobytes()


def _speaker_is_unidentified(speaker_id: object, speaker_name: object) -> bool:
    """Unidentified means no usable speaker assignment, not an unknown name."""
    return (
        type(speaker_id) is not int or speaker_id <= 0
        or str(speaker_name or "").strip().lower() == "unidentified speaker"
    )


def _speaker_transition(old_id: object, old_name: object,
                        new_id: object, new_name: object) -> str:
    old_unknown = _speaker_is_unidentified(old_id, old_name)
    new_unknown = _speaker_is_unidentified(new_id, new_name)
    if old_unknown and not new_unknown:
        return "unidentified_to_labeled"
    if not old_unknown and new_unknown:
        return "labeled_to_unidentified"
    if old_unknown and new_unknown:
        return "unidentified_to_other_unidentified"
    return "labeled_to_different_labeled"


def _timing_bucket(state: dict, elapsed_ms: float) -> dict:
    """Group streamed captions into elapsed-wall-time windows."""
    interval = state["timing_interval_s"]
    index = int(max(0, elapsed_ms) // (interval * 1000))
    if index not in state["timing_buckets"]:
        state["timing_buckets"][index] = {
            "events": Counter(), "preview_lags": [],
            "confirmation_lags": [], "new_segments": 0,
            "speaker_label_changes": 0,
            "confirmed_text_changes": 0,
            "max_transport_gap_ms": 0.0,
        }
    return state["timing_buckets"][index]


def _clock_wait_seconds(
    start: float, samples_sent: int, sample_rate: int, speed: float,
    now: float,
) -> float:
    """Wait to an absolute media-time deadline, not 0.2s per send."""
    if sample_rate <= 0 or speed <= 0:
        raise ValueError("Invalid replay pacing")
    return max(0.0, start + samples_sent / (sample_rate * speed) - now)


def _source_end_lag(elapsed_ms: float, end_ms: object) -> float | None:
    """Apparent delay since reported utterance end, not word-level latency."""
    if type(end_ms) not in (int, float) or not math.isfinite(end_ms):
        return None
    delta = elapsed_ms - end_ms
    return round(delta, 1) if delta >= 0 else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(0.95 * len(ordered)) - 1], 1)


def summarize_timing(state: dict, elapsed_ms: float) -> list[dict]:
    """Report whether apparent caption delay grows over a long session."""
    width = state["timing_interval_s"]
    periods = max(1, math.ceil(max(0, elapsed_ms) / (1000 * width)))
    result = []
    for index in range(periods):
        bucket = state["timing_buckets"].get(index) or {}
        previews = bucket.get("preview_lags", [])
        confirms = bucket.get("confirmation_lags", [])
        result.append({
            "from_s": width * index,
            "to_s": min(width * (index + 1), round(elapsed_ms / 1000, 1)),
            "events": dict(bucket.get("events", {})),
            "new_confirmed_segments": bucket.get("new_segments", 0),
            "speaker_label_changes": bucket.get("speaker_label_changes", 0),
            "confirmed_text_changes": bucket.get("confirmed_text_changes", 0),
            "preview_lag_samples": len(previews),
            "preview_lag_median_ms": (
                round(median(previews), 1) if previews else None),
            "preview_lag_p95_ms": _p95(previews),
            "confirmation_lag_samples": len(confirms),
            "confirmation_lag_median_ms": (
                round(median(confirms), 1) if confirms else None),
            "confirmation_lag_p95_ms": _p95(confirms),
            "max_audio_transport_gap_ms": round(
                bucket.get("max_transport_gap_ms", 0.0), 1),
        })
    return result


def _speaker_switch_age_bucket(
    elapsed_ms: float, segment_end_ms: object,
    sent_samples: int | None,
) -> str:
    """Age of audio evidence when a displayed speaker label was changed."""
    if type(segment_end_ms) not in (int, float) or not math.isfinite(segment_end_ms):
        return "timestamp_unavailable"
    media_elapsed_ms = (
        sent_samples * 1000 / SAMPLE_RATE
        if type(sent_samples) is int else elapsed_ms
    )
    age_s = max(0.0, (media_elapsed_ms - segment_end_ms) / 1000)
    if age_s < 15:
        return "0_to_15_s"
    if age_s < 60:
        return "15_to_60_s"
    if age_s < 120:
        return "60_to_120_s"
    return "120_s_or_more"


def capture_event(state: dict, message: dict, elapsed_ms: float,
                  *, sent_samples: int | None = None) -> bool:
    """Count events without writing private transcript strings to disk."""
    kind = str(message.get("type", "unknown"))
    state["events"][kind] += 1
    bucket = _timing_bucket(state, elapsed_ms)
    bucket["events"][kind] += 1
    if kind == "partial":
        revision = message.get("revision")
        if not isinstance(revision, int) or revision <= state["last_partial_revision"]:
            state["flags"].append("nonmonotonic_partial_revision")
        else:
            state["last_partial_revision"] = revision
        text = str(message.get("text", "") or "").strip()
        if text and state["first_preview_ms"] is None:
            state["first_preview_ms"] = round(elapsed_ms, 1)
        if text:
            lag = _source_end_lag(elapsed_ms, message.get("end_ms"))
            if lag is not None:
                bucket["preview_lags"].append(lag)
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
        prior_names = state["segment_speaker_names"]
        prior_text = state["segment_texts"]
        updated = {}
        updated_names = {}
        updated_text = {}
        for segment in incoming:
            sid = segment.get("id")
            if type(sid) is not int:
                state["flags"].append("invalid_segment_id")
                continue
            voice = segment.get("speaker_id")
            name = segment.get("speaker_name")
            text = str(segment.get("text") or "")
            updated[sid] = voice
            updated_names[sid] = name
            updated_text[sid] = text
            if sid not in previous:
                bucket["new_segments"] += 1
                lag = _source_end_lag(elapsed_ms, segment.get("end_ms"))
                if lag is not None:
                    bucket["confirmation_lags"].append(lag)
            if sid in previous:
                if previous[sid] != voice:
                    state["speaker_reassignments"] += 1
                    bucket["speaker_label_changes"] += 1
                    state["speaker_change_counts"][sid] += 1
                    transition = _speaker_transition(
                        previous[sid], prior_names.get(sid), voice, name,
                    )
                    state["speaker_change_types"][transition] += 1
                    if transition == "labeled_to_different_labeled":
                        age_bucket = _speaker_switch_age_bucket(
                            elapsed_ms, segment.get("end_ms"), sent_samples,
                        )
                        state["labeled_switch_age_buckets"][age_bucket] += 1
                        if (type(previous[sid]) is int
                                and type(voice) is int):
                            state["labeled_switch_edges"][
                                (previous[sid], voice)
                            ] += 1
                if sid in prior_text and prior_text[sid] != text:
                    state["confirmed_text_changes"] += 1
                    bucket["confirmed_text_changes"] += 1
                    state["changed_text_ids"].add(sid)
        if len(updated) != len(incoming):
            state["flags"].append("duplicate_segment_ids")
        disappeared = set(previous) - set(updated)
        state["confirmed_segment_disappearances"] += len(disappeared)
        # Text is held in memory for comparison, never persisted in reports.
        state["segment_texts"] = updated_text
        state["segment_speakers"] = updated
        state["segment_speaker_names"] = updated_names
        state["final_segment_count"] = len(updated)
        state["max_visible_segments"] = max(
            state["max_visible_segments"], len(updated)
        )
    elif kind == "feature_warning":
        feature = str(message.get("feature") or "unspecified")[:80]
        state["feature_warning_counts"][feature] += 1
        sample = {
            "feature": feature,
            "message": str(message.get("message") or "")[:240],
        }
        # Keep only a few distinct diagnostic messages, never audio or
        # recognized transcript text. Reports remain in gitignored storage.
        if (sample not in state["feature_warning_samples"]
                and len(state["feature_warning_samples"]) < 12):
            state["feature_warning_samples"].append(sample)
    elif kind == "audio_progress":
        accepted = message.get("samples")
        if (type(accepted) is int and type(sent_samples) is int
                and 0 <= accepted <= sent_samples):
            gap_ms = (sent_samples - accepted) * 1000 / SAMPLE_RATE
            bucket["max_transport_gap_ms"] = max(
                bucket["max_transport_gap_ms"], gap_ms)
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


def new_capture(interval_seconds: int = 300) -> dict:
    if not 30 <= interval_seconds <= 1800:
        raise ValueError("Timing interval must be within 30..1800 seconds")
    return {
        "timing_interval_s": interval_seconds, "timing_buckets": {},
        "events": Counter(), "flags": [], "errors": [],
        "first_preview_ms": None, "first_confirmed_ms": None,
        "last_partial_revision": 0, "last_transcript_version": 0,
        "last_preview_chars": 0, "clear_events": 0,
        "segment_speakers": {}, "segment_speaker_names": {},
        "segment_texts": {},
        "speaker_reassignments": 0,
        "speaker_change_types": Counter(),
        "speaker_change_counts": Counter(),
        "labeled_switch_age_buckets": Counter(),
        "labeled_switch_edges": Counter(),
        "confirmed_text_changes": 0, "changed_text_ids": set(),
        "confirmed_segment_disappearances": 0,
        "max_visible_segments": 0,
        "feature_warning_counts": Counter(), "feature_warning_samples": [],
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

    counters = new_capture(args.interval_seconds)
    sent_samples = 0
    sent_sequence = 0
    first_send = None
    replay_clock_ms = []
    send_elapsed_ms = None
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
            "translation_target": opening.get("translation_target"),
        }

        async def receive():
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                elapsed = 0.0 if first_send is None else (
                    time.perf_counter() - first_send) * 1000
                if capture_event(
                        counters, msg, elapsed, sent_samples=sent_samples):
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
                        # Absolute-deadline pacing prevents per-frame scheduler
                        # oversleep from accumulating over long Windows runs.
                        now = time.perf_counter()
                        # Measure lateness against the START of the frame
                        # just sent, not the next frame's future deadline.
                        frame_start_samples = sent_samples - len(pcm)
                        drift_ms = (
                            now - first_send
                            - frame_start_samples / (SAMPLE_RATE * args.speed)
                        ) * 1000
                        replay_clock_ms.append(drift_ms)
                        pause_s = _clock_wait_seconds(
                            first_send, sent_samples, SAMPLE_RATE, args.speed, now)
                        if pause_s > 0:
                            await asyncio.sleep(pause_s)
                if args.loops > 1:
                    print(
                        f"[{loop_index + 1}/{args.loops}] "
                        f"{sent_samples / SAMPLE_RATE:.1f}s PCM sent "
                        f"confirmed={counters['final_segment_count']} "
                        f"warnings={sum(counters['feature_warning_counts'].values())}",
                        flush=True,
                    )
            # Capture the end of PCM transmission BEFORE stop/ASR
            # finalization. Terminal processing must not count as pacing drift.
            send_elapsed_ms = round(
                (time.perf_counter() - first_send) * 1000, 1)
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
    playback_ms = round(
        (time.perf_counter() - first_send) * 1000, 1
    ) if first_send is not None else None
    summary = {
        "schema": SCHEMA,
        "source_audio": str(source),
        "source_audio_duration_s": round(len(wave) / SAMPLE_RATE, 2),
        "loops": args.loops, "gap_ms": args.gap_ms, "playback_speed": args.speed,
        "real_time_latency_evaluable": args.speed == 1.0,
        "sent_frames": sent_sequence, "sent_samples": sent_samples,
        "sent_audio_duration_s": round(sent_samples / SAMPLE_RATE, 2),
        "total_wall_ms": elapsed_ms,
        "max_late_send_ms": round(max(replay_clock_ms, default=0.0), 1),
        "max_early_send_ms": round(max(
            (-value for value in replay_clock_ms), default=0.0), 1),
        "replay_pacing_method": "absolute_monotonic_deadline",
        "expected_replay_wall_ms": round(
            sent_samples * 1000 / (SAMPLE_RATE * args.speed), 1),
        "wall_minus_media_ms": (
            round(playback_ms -
                  sent_samples * 1000 / (SAMPLE_RATE * args.speed), 1)
            if playback_ms is not None else None),
        "sender_pacing_drift_ms": (
            round(send_elapsed_ms -
                  sent_samples * 1000 / (SAMPLE_RATE * args.speed), 1)
            if send_elapsed_ms is not None else None),
        "sender_timing_measurable": True,
        "timing_interval_s": args.interval_seconds,
        "timing_by_elapsed_window": summarize_timing(counters, elapsed_ms),
        "timestamp_lag_note": (
            "Apparent lag compares wall-clock event reception with ASR "
            "utterance end timestamps. Absolute-time PCM pacing reduces "
            "scheduler accumulation. This does not measure word-by-word "
            "latency or certify transcript completeness."
        ),
        "server_ready": ready,
        "events": dict(counters["events"]),
        "first_preview_wall_ms": counters["first_preview_ms"],
        "first_confirmed_wall_ms": counters["first_confirmed_ms"],
        "final_confirmed_segments": counters["final_segment_count"],
        "speaker_reassignments": counters["speaker_reassignments"],
        "speaker_reassignment_types": dict(counters["speaker_change_types"]),
        "labeled_switch_age_buckets": dict(counters["labeled_switch_age_buckets"]),
        "labeled_switch_edges_top10": [
            {"from_id": old, "to_id": new, "changes": count}
            for (old, new), count in counters["labeled_switch_edges"].most_common(10)
        ],
        "speaker_change_hotspots_top10": [
            {"segment_id": sid, "changes": count}
            for sid, count in counters["speaker_change_counts"].most_common(10)
        ],
        "final_assigned_speaker_count": len({
            voice for sid, voice in counters["segment_speakers"].items()
            if not _speaker_is_unidentified(
                voice, counters["segment_speaker_names"].get(sid)
            )
        }),
        "final_unidentified_segment_count": sum(
            _speaker_is_unidentified(
                voice, counters["segment_speaker_names"].get(sid)
            ) for sid, voice in counters["segment_speakers"].items()
        ),
        "segments_with_speaker_changes": len(counters["speaker_change_counts"]),
        "max_speaker_changes_one_segment": max(
            counters["speaker_change_counts"].values(), default=0),
        "confirmed_text_revisions": counters["confirmed_text_changes"],
        "segments_with_text_revisions": len(counters["changed_text_ids"]),
        "confirmed_segment_disappearances": counters["confirmed_segment_disappearances"],
        "max_visible_confirmed_segments": counters["max_visible_segments"],
        "feature_warning_counts": dict(counters["feature_warning_counts"]),
        "feature_warning_samples": counters["feature_warning_samples"],
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
    if (summary["sender_pacing_drift_ms"] is not None
            and summary["sender_pacing_drift_ms"] > 2000):
        summary["flags"].append("sender_pacing_more_than_2s_late")
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
    # An absent translate_to query parameter inherits MEETING_TRANSLATE_TO
    # (which defaults to English). An explicit empty value matches the
    # browser's "No translation" option and isolates ASR/diarization testing.
    parser.add_argument("--url", default=DEFAULT_WS_URL)
    parser.add_argument("--origin", default="http://localhost:5173")
    parser.add_argument("--loops", type=int, default=1,
                        help="Repeat same source for endurance only, not accuracy")
    parser.add_argument("--gap-ms", type=int, default=500)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Only speed=1.0 gives real-time timing")
    parser.add_argument("--final-timeout", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=int, default=300,
                        help="Group progress and caption lag every 5 minutes")
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

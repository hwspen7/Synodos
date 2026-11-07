"""Locate which caption/identity evidence changed between paired live AMI replays.

Only anonymous time geometry is compared. Speaker IDs are intentionally NOT
compared across replays, since valid label permutations are possible. This is
forensic attribution, not a DER/WER score or a replacement for listening.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STEP_MS = 100
MAX_SECONDS = 14400


def _timeline(replay: dict) -> list[dict]:
    diag = replay.get("final_cluster_diagnostics") or {}
    timeline = diag.get("anonymous_timeline")
    if not isinstance(timeline, dict) or timeline.get("truncated"):
        raise ValueError("Incomplete anonymous speaker timeline")
    rows = timeline.get("segments")
    if (not isinstance(rows, list)
            or timeline.get("total_segments") != len(rows)
            or replay.get("final_confirmed_segments") != len(rows)):
        raise ValueError("Final transcript count differs from complete timeline")
    return rows


def _paired_duration(a: dict, b: dict) -> int:
    required = ("source_audio_sha256", "source_audio_duration_s",
                "loops", "playback_speed")
    for key in required:
        if a.get(key) != b.get(key) or a.get(key) is None:
            raise ValueError("Replays must have paired " + key)
    if not isinstance(a["source_audio_sha256"], str) or not a["source_audio_sha256"]:
        raise ValueError("Missing audio checksum")
    if a["loops"] != 1 or a["playback_speed"] != 1.0:
        raise ValueError("Require one 1x real-time playback")
    seconds = a["source_audio_duration_s"]
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not 0 < seconds <= MAX_SECONDS):
        raise ValueError("Invalid audio duration")
    if (str(a.get("source_audio") or "").replace("\\", "/").split("/")[-1]
            != str(b.get("source_audio") or "").replace("\\", "/").split("/")[-1]):
        raise ValueError("Source WAV names differ")
    return round(seconds * 1000)


def _frame_activity(rows: list[dict], total_ms: int) -> dict[str, list[bool]]:
    frames = (total_ms + STEP_MS - 1) // STEP_MS
    diff = {name: [0] * (frames + 1) for name in
            ("caption", "positive_speaker", "embedding")}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed timeline row")
        start, end = row.get("start_ms"), row.get("end_ms")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= total_ms + STEP_MS):
            raise ValueError("Invalid anonymous segment timestamps")
        left = (start + STEP_MS - 1) // STEP_MS
        right = min(frames, (end + STEP_MS - 1) // STEP_MS)
        fields = ["caption"]
        if (type(row.get("published_speaker_id")) is int
                and row["published_speaker_id"] > 0):
            fields.append("positive_speaker")
        if row.get("embedding_available") is True:
            fields.append("embedding")
        for name in fields:
            diff[name][left] += 1
            diff[name][right] -= 1
    values = {}
    for name, edges in diff.items():
        count = 0
        series = []
        for item in edges[:-1]:
            count += item
            series.append(count > 0)
        values[name] = series
    return values


def _intervals(mask: list[bool], limit: int = 20) -> dict:
    runs = []
    start = None
    for i in range(len(mask) + 1):
        active = mask[i] if i < len(mask) else False
        if active and start is None:
            start = i
        if not active and start is not None:
            runs.append({
                "start_ms": start * STEP_MS,
                "end_ms": i * STEP_MS,
                "duration_ms": (i - start) * STEP_MS,
            })
            start = None
    runs.sort(key=lambda r: (-r["duration_ms"], r["start_ms"]))
    return {
        "total_seconds": round(sum(r["duration_ms"] for r in runs) / 1000, 2),
        "interval_count": len(runs),
        "longest_intervals": runs[:limit],
    }


def compare_replays(baseline: dict, candidate: dict) -> dict:
    total_ms = _paired_duration(baseline, candidate)
    before_rows, after_rows = _timeline(baseline), _timeline(candidate)
    before = _frame_activity(before_rows, total_ms)
    after = _frame_activity(after_rows, total_ms)
    changes = {}
    for key in ("caption", "positive_speaker", "embedding"):
        changes["lost_" + key] = _intervals([
            a and not b for a, b in zip(before[key], after[key])
        ])
        changes["gained_" + key] = _intervals([
            b and not a for a, b in zip(before[key], after[key])
        ])
    changes["caption_to_unidentified"] = _intervals([
        before["positive_speaker"][i]
        and after["caption"][i] and not after["positive_speaker"][i]
        for i in range(len(before["caption"]))
    ])
    return {
        "schema": "meeting-ami-anonymous-replay-delta-v1",
        "source_audio_sha256": baseline["source_audio_sha256"],
        "frame_step_ms": STEP_MS,
        "duration_ms": total_ms,
        "baseline_segments": len(before_rows),
        "candidate_segments": len(after_rows),
        "changes": changes,
        "interpretation": (
            "lost_caption identifies intervals that had a live subtitle in the "
            "baseline but none in the candidate, not proven missing words. "
            "lost_positive_speaker includes both missing captions and newly "
            "unidentified captions. Other speaker IDs are not directly "
            "compared across replays because labels can be permuted. "
            "Review original audio and the paired AMI DER gate separately."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    after = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_replays(before, after)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

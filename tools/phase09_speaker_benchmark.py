"""Score real speaker diarization against manually annotated speaker turns.

Inputs are JSON arrays of {start_ms, end_ms, speaker} for reference and
{start_ms, end_ms, speaker_id} for hypothesis. Speaker IDs are permutation
invariant; unknown speakers count as missed speech rather than a correct ID.

The scoring is time-based and approximate (100 ms frames by default).
Overlapped reference speech is explicitly counted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path


def _intervals(records: list[dict], label: str) -> list[tuple[int, int, str]]:
    intervals = []
    for value in records:
        start, end = int(value["start_ms"]), int(value["end_ms"])
        name = str(value.get(label, "")).strip()
        if start < 0 or end <= start:
            raise ValueError("Invalid time interval")
        if not name:
            raise ValueError("Missing speaker identity")
        if label == "speaker_id" and (name.startswith("-") or name == "0"):
            continue
        intervals.append((start, end, name))
    return intervals


def _assignment(weights: dict[tuple[str, str], int]) -> dict[str, str]:
    """Maximum weight one-to-one matching using exact bitmask DP.

    Tests use a bounded group size; rejecting larger sets is preferable to
    silently producing an incorrect score.
    """
    predicted = sorted({p for _, p in weights})
    reference = sorted({r for r, _ in weights})
    if len(reference) > 12 or len(predicted) > 12:
        raise ValueError("At most 12 distinct speakers per comparison")
    @lru_cache(None)
    def solve(index: int, used: int) -> tuple[int, tuple[tuple[str, str], ...]]:
        if index == len(predicted):
            return 0, ()
        score, pairs = solve(index + 1, used)
        for j, reference_id in enumerate(reference):
            if used & (1 << j):
                continue
            add = weights.get((reference_id, predicted[index]), 0)
            later_score, later_pairs = solve(index + 1, used | (1 << j))
            if add + later_score > score:
                score = add + later_score
                pairs = later_pairs + ((predicted[index], reference_id),)
        return score, pairs
    return dict(solve(0, 0)[1])


def evaluate(reference: list[dict], hypothesis: list[dict], step_ms: int = 100) -> dict:
    if not 20 <= step_ms <= 500:
        raise ValueError("Frame step must be 20..500 milliseconds")
    ref = _intervals(reference, "speaker")
    pred = _intervals(hypothesis, "speaker_id")
    if not ref:
        raise ValueError("A nonempty manually labeled reference is required")
    duration = max((b for _, b, _ in ref + pred), default=0)
    frames = []
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for time_ms in range(0, duration, step_ms):
        ref_speakers = {name for a, b, name in ref if a <= time_ms < b}
        pred_speakers = {name for a, b, name in pred if a <= time_ms < b}
        frames.append((ref_speakers, pred_speakers))
        for a in ref_speakers:
            for b in pred_speakers:
                weights[(a, b)] += 1
    mapping = _assignment(weights)
    reference_time = 0
    miss = false_alarm = confusion = identified = 0
    for real, detected in frames:
        r, h = len(real), len(detected)
        reference_time += r
        matches = len(real.intersection({mapping.get(p, "") for p in detected}))
        identified += matches
        miss += max(0, r - h)
        false_alarm += max(0, h - r)
        confusion += max(0, min(r, h) - matches)
    total_error = miss + false_alarm + confusion
    return {
        "der": round(total_error / reference_time, 5) if reference_time else None,
        "missed_speaker_time": round(miss / reference_time, 5) if reference_time else None,
        "false_alarm_time": round(false_alarm / reference_time, 5) if reference_time else None,
        "confusion_time": round(confusion / reference_time, 5) if reference_time else None,
        "reference_speaker_seconds": round(reference_time * step_ms / 1000, 2),
        "speaker_mapping": mapping,
        "frame_step_ms": step_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-der", type=float, default=0.20)
    options = parser.parse_args()
    result = evaluate(
        json.loads(options.reference.read_text(encoding="utf-8")),
        json.loads(options.hypothesis.read_text(encoding="utf-8")),
    )
    body = json.dumps(result, indent=2)
    print(body)
    if options.output:
        options.output.write_text(body + "\n", encoding="utf-8")
    return 0 if result["der"] is not None and result["der"] <= options.max_der else 1


if __name__ == "__main__":
    raise SystemExit(main())

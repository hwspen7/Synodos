"""Offline CAM++ probe on AMI *human-reference solo* speech.

This is an oracle-context diagnostic. Human reference annotations are
used to select audio; there is no production speaker discovery. Compare
whether distinct annotated voices are acoustically separable in CAM++
embedding space before modifying live clustering or speaker enrollment.

Output contains only bounded timestamps, sample counts and cosine
statistics. Never persist waveforms, embeddings or recognized words.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from speech.diarization.speaker_backends import unit_embedding
from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.validation.session_quality_benchmark import load_wav

SAMPLE_RATE = 16000
MIN_WINDOW_MS = 2000
MAX_WINDOW_MS = 3200
EDGE_MARGIN_MS = 200
MAX_WINDOWS_PER_SPEAKER = 6


def reference_solo_runs(reference: list[dict], duration_ms: int) -> dict[str, list[tuple[int, int]]]:
    """Exact millisecond intervals with one unique annotated voice active."""
    events: dict[int, list[tuple[str, int]]] = {}
    for turn in reference:
        start, end, speaker = (
            turn["start_ms"], turn["end_ms"], turn["speaker"]
        )
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= duration_ms
                or not isinstance(speaker, str) or not speaker):
            raise ValueError("Invalid AMI annotation interval")
        events.setdefault(start, []).append((speaker, 1))
        events.setdefault(end, []).append((speaker, -1))
    active: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    previous = None
    for timestamp in sorted(events):
        alive = [name for name, count in active.items() if count > 0]
        if previous is not None and timestamp > previous and len(alive) == 1:
            name = alive[0]
            group = intervals.setdefault(name, [])
            if group and group[-1][1] == previous:
                group[-1] = (group[-1][0], timestamp)
            else:
                group.append((previous, timestamp))
        for name, delta in events[timestamp]:
            n = active.get(name, 0) + delta
            if n <= 0:
                active.pop(name, None)
            else:
                active[name] = n
        previous = timestamp
    return intervals


def select_solo_windows(
    runs: dict[str, list[tuple[int, int]]], *,
    maximum: int = MAX_WINDOWS_PER_SPEAKER,
) -> dict[str, list[tuple[int, int]]]:
    """Long solo runs first; at most two non-overlapping windows per run."""
    if not 1 <= maximum <= 12:
        raise ValueError("Invalid solo-window budget")
    result = {}
    for speaker, intervals in sorted(runs.items()):
        selected = []
        for start, end in sorted(
            intervals, key=lambda pair: (-(pair[1] - pair[0]), pair[0]),
        ):
            left, right = start + EDGE_MARGIN_MS, end - EDGE_MARGIN_MS
            if right - left < MIN_WINDOW_MS:
                continue
            # Include short but clean 2s audio rather than extending
            # the interval into another person's speech.
            length = min(MAX_WINDOW_MS, (right - left) // 2)
            if length >= MIN_WINDOW_MS:
                selected.append((left, left + length))
                if len(selected) < maximum:
                    selected.append((right - length, right))
            else:
                selected.append((left, min(right, left + MAX_WINDOW_MS)))
            if len(selected) >= maximum:
                break
        result[speaker] = selected[:maximum]
    return result


def scalar_similarities(
    extracted: dict[str, list[np.ndarray]],
    selected_windows: dict[str, list[tuple[int, int]]],
) -> dict:
    """Summaries from normalized embeddings; no audio vectors exported."""
    embeddings = {}
    for label, vectors in extracted.items():
        checked = [unit_embedding(vector) for vector in vectors]
        if any(vector is None for vector in checked):
            raise ValueError("Invalid standalone speaker embedding")
        embeddings[label] = checked
    metadata = []
    for speaker in sorted(selected_windows):
        vectors = embeddings.get(speaker, [])
        windows = selected_windows[speaker]
        if len(vectors) > len(windows):
            raise ValueError("More voiceprints than selected AMI windows")
        within = [
            float(np.dot(left, right))
            for index, left in enumerate(vectors)
            for right in vectors[index + 1:]
        ]
        metadata.append({
            "reference_speaker": speaker,
            "selected_windows": [
                {"start_ms": a, "end_ms": b}
                for a, b in windows
            ],
            "successful_voiceprints": len(vectors),
            "within_speaker_cosine": {
                "pairs": len(within),
                "minimum": round(min(within), 5) if within else None,
                "median": round(float(np.median(within)), 5) if within else None,
                "maximum": round(max(within), 5) if within else None,
            },
        })
    between = []
    speakers = sorted(embeddings)
    for index, left in enumerate(speakers):
        for right in speakers[index + 1:]:
            scores = [
                float(np.dot(a, b))
                for a in embeddings[left]
                for b in embeddings[right]
            ]
            if scores:
                between.append({
                    "reference_speaker_a": left,
                    "reference_speaker_b": right,
                    "pair_count": len(scores),
                    "maximum_cosine": round(max(scores), 5),
                    "median_cosine": round(float(np.median(scores)), 5),
                    "minimum_cosine": round(min(scores), 5),
                })
    return {
        "speakers": metadata,
        "different_speaker_pair_cosines": between,
        "warning": (
            "Human AMI annotations selected clean speech: oracle-context "
            "evidence, not live diarization accuracy. Similarity overlap "
            "requires held-out calibration; no automatic identity change."
        ),
    }


def analyze_solo(
    reference: list[dict], samples: np.ndarray, embedding_backend,
    *, cluster_analysis: bool = False,
) -> dict:
    if samples.ndim != 1 or len(samples) != 240 * SAMPLE_RATE:
        raise ValueError("Require frozen 240s mono 16kHz source audio")
    runs = reference_solo_runs(reference, 240000)
    windows = select_solo_windows(runs)
    data = {}
    failures = []
    for speaker, spans in windows.items():
        succeeded = []
        for start, end in spans:
            audio = np.ascontiguousarray(
                samples[start * 16:end * 16], dtype=np.float32,
            )
            if (len(audio) < MIN_WINDOW_MS * 16 or
                    not np.isfinite(audio).all()):
                failures.append({
                    "reference_speaker": speaker,
                    "start_ms": start, "end_ms": end,
                    "reason": "invalid_audio",
                })
                continue
            try:
                vector = unit_embedding(embedding_backend.embed(audio))
            except Exception:
                vector = None
            if vector is None:
                failures.append({
                    "reference_speaker": speaker,
                    "start_ms": start, "end_ms": end,
                    "reason": "embedding_unavailable",
                })
                continue
            succeeded.append(vector)
        data[speaker] = succeeded
    result = scalar_similarities(data, windows)
    if cluster_analysis:
        from tools.evaluation.ami.ami_oracle_cluster_eval import evaluate_oracle_clusters
        result["clustering_evaluation"] = evaluate_oracle_clusters(data)
    result["failed_windows"] = failures
    result["usable_reference_speakers"] = sum(
        bool(x) for x in data.values()
    )
    result["source_duration_s"] = 240
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("benchmark/fixed195/ami_shadow10"),
    )
    parser.add_argument("--meeting", required=True,
                        choices=[m for m, _ in SELECTED])
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cluster-analysis", action="store_true",
        help="Evaluate unsupervised clustering on oracle-clean voiceprints",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output.resolve()
    private = Path("benchmark/fixed195").resolve()
    if (not root.is_relative_to(private)
            or not destination.is_relative_to(root)
            or destination.suffix.lower() != ".json"
            or destination.exists()):
        parser.error("Write a new private JSON under gitignored Fixed195")
    # Validate the complete human-reference, source audio and live replay.
    audit = score_case(root, args.meeting, args.replay)
    if audit["meeting"] != args.meeting:
        raise ValueError("Unexpected meeting provenance")
    case = json.loads(
        (root / "cases" / (args.meeting + ".json")).read_text(encoding="utf-8")
    )
    audio = (root / case["audio"]).resolve()
    if not audio.is_relative_to(root):
        raise ValueError("Audio escaped private AMI root")
    samples = load_wav(audio)
    from core.config import Settings
    from inference.models import ModelManager
    from speech.diarization.speaker_backends import CamPlusEmbedding
    manager = ModelManager(Settings())
    try:
        report = analyze_solo(
            case["reference_turns"], samples, CamPlusEmbedding(manager),
            cluster_analysis=args.cluster_analysis,
        )
    finally:
        manager.close()
    report.update({
        "schema": "meeting-ami-oracle-solo-camplus-probe-v1",
        "meeting": args.meeting,
        "source_audio_sha256": audit["source_audio_sha256"],
        "reference_source_sha256": audit["reference_source_sha256"],
        "baseline_replay_sha256": audit["replay_sha256"],
        "model": "production/camplus",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

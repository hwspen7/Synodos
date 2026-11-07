"""Offline continuous-window CAM++ audit against pinned AMI reference RTTM.

This deliberately samples audio at the live acoustic timeline's nominal
2.0s window / 1.5s stride, WITHOUT using RTTM to choose any window.
RTTM is consulted only AFTER extraction and unsupervised clustering,
for evaluation. It does not simulate live task backpressure, late
arrivals, VAD, ASR, or the exact 88 transcript-bound embeddings.

No audio, raw voiceprints, or recognized words are written to reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from speech.diarization.speaker_backends import unit_embedding
from speech.diarization.spherical_cluster import spherical_speaker_candidates
from tools.evaluation.ami.ami_oracle_cluster_eval import (
    provisional_isolated_members, score_reference_partition,
)
from tools.evaluation.ami.ami_cluster_substructure import explore_speaker_substructure
from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.validation.session_quality_benchmark import load_wav

SAMPLE_RATE = 16000
WINDOW_MS = 2000
STRIDE_MS = 1500
MIN_RMS = 0.001
PURITY_MIN_FRACTION = 0.80
OTHER_MAX_FRACTION = 0.10
MAX_WINDOWS = 256


def reference_window_coverage(reference: list[dict], start_ms: int,
                              end_ms: int) -> dict[str, int]:
    """Coverage for each person, unioning multiple intervals from that person."""
    if not 0 <= start_ms < end_ms:
        raise ValueError("Invalid reference window")
    groups: dict[str, list[tuple[int, int]]] = {}
    for row in reference:
        a = max(start_ms, int(row["start_ms"]))
        b = min(end_ms, int(row["end_ms"]))
        if b > a:
            groups.setdefault(str(row["speaker"]), []).append((a, b))
    totals = {}
    for speaker, pieces in groups.items():
        pieces.sort()
        covered = 0
        active_left = active_right = None
        for left, right in pieces:
            if active_left is None:
                active_left, active_right = left, right
            elif left <= active_right:
                active_right = max(active_right, right)
            else:
                covered += active_right - active_left
                active_left, active_right = left, right
        if active_left is not None:
            covered += active_right - active_left
        totals[speaker] = covered
    return totals


def reference_window_classification(reference: list[dict],
                                    start_ms: int, end_ms: int) -> dict:
    """Post-hoc reference label: pure, mixed, or unclassifiable."""
    duration = end_ms - start_ms
    counts = reference_window_coverage(reference, start_ms, end_ms)
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if not ordered:
        return {
            "reference_context": "unannotated",
            "reference_speaker": None, "dominant_fraction": 0.0,
            "other_max_fraction": 0.0,
        }
    leader, amount = ordered[0]
    other_max = ordered[1][1] if len(ordered) > 1 else 0
    dominant = round(amount / duration, 5)
    other = round(other_max / duration, 5)
    pure = (
        amount / duration >= PURITY_MIN_FRACTION
        and other_max / duration <= OTHER_MAX_FRACTION
    )
    context = "pure" if pure else (
        "multi_voice" if len(ordered) > 1 and other_max > duration * OTHER_MAX_FRACTION
        else "partial_or_boundary"
    )
    return {
        "reference_context": context,
        "reference_speaker": leader if pure else None,
        "dominant_fraction": dominant,
        "other_max_fraction": other,
    }


def extract_continuous_windows(samples: np.ndarray, backend) -> tuple[list[np.ndarray], list[dict]]:
    """Extract on a fixed grid independent of ground-truth RTTM."""
    audio = np.asarray(samples, dtype=np.float32)
    if (audio.ndim != 1 or audio.size < WINDOW_MS * 16
            or audio.size > 240 * SAMPLE_RATE
            or not np.isfinite(audio).all()):
        raise ValueError("Expected finite 2..240s 16kHz mono audio")
    window = WINDOW_MS * 16
    stride = STRIDE_MS * 16
    vectors = []
    events = []
    for start in range(0, audio.size - window + 1, stride):
        if len(events) >= MAX_WINDOWS:
            raise ValueError("Continuous window budget exceeded")
        segment = np.ascontiguousarray(audio[start:start + window])
        rms = float(np.sqrt(np.mean(segment * segment)))
        event = {
            "start_ms": start // 16,
            "end_ms": (start + window) // 16,
            "rms": round(rms, 6),
        }
        if rms < MIN_RMS:
            event["status"] = "rms_rejected"
        else:
            try:
                emb = unit_embedding(backend.embed(segment))
            except Exception:
                emb = None
            if emb is None:
                event["status"] = "embedding_unavailable"
            else:
                event["status"] = "embedded"
                event["vector_index"] = len(vectors)
                vectors.append(emb)
        events.append(event)
    return vectors, events


def audit_continuous_windows(
    vectors: list[np.ndarray], events: list[dict],
    reference: list[dict],
) -> dict:
    """Cluster ALL successful windows and label them AFTER inference."""
    result = {
        "method": "offline_continuous_2s_1p5s",
        "nominal_window_ms": WINDOW_MS,
        "nominal_stride_ms": STRIDE_MS,
        "min_rms": MIN_RMS,
        "pure_dominant_fraction": PURITY_MIN_FRACTION,
        "pure_other_max_fraction": OTHER_MAX_FRACTION,
        "total_windows": len(events),
        "successful_embeddings": len(vectors),
        "rms_rejected": sum(x["status"] == "rms_rejected" for x in events),
        "embedding_unavailable": sum(
            x["status"] == "embedding_unavailable" for x in events
        ),
        "true_speakers_from_rttm": len({
            row["speaker"] for row in reference
        }),
        "warnings": [
            "No RTTM annotations selected extraction or model cluster count.",
            "Reference 'pure' means >=80% annotated time for one person "
            "and <=10% for any other person, not verified audio purity.",
            "This is not the original live scheduling, ASR-aligned embedding "
            "stream, overlap transcription, or DER.",
            "Candidate singleton groups are diagnostic only; no production "
            "speaker identity changes or enrollment.",
        ],
    }
    pure_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    evaluations = []
    for event in events:
        posthoc = reference_window_classification(
            reference, event["start_ms"], event["end_ms"]
        )
        context_counts[posthoc["reference_context"]] += 1
        if event["status"] != "embedded":
            continue
        if posthoc["reference_speaker"] is not None:
            pure_counts[posthoc["reference_speaker"]] += 1
        evaluations.append((event["vector_index"], posthoc))
    result["reference_context_window_counts"] = dict(context_counts)
    result["pure_voiceprint_counts"] = {
        speaker: pure_counts.get(speaker, 0)
        for speaker in sorted({row["speaker"] for row in reference})
    }
    if len(vectors) < 4 or len(vectors) > MAX_WINDOWS:
        result.update({
            "available": False,
            "reason": "insufficient_or_over_budget_embeddings",
        })
        return result
    array = np.stack(vectors).astype(np.float32)
    candidate = spherical_speaker_candidates(array)
    result["spherical_candidate"] = candidate.details
    if candidate.labels is None:
        result.update({
            "available": False, "reason": "spherical_candidate_abstained",
        })
        return result
    provisional, flags = provisional_isolated_members(
        array, candidate.labels
    )
    substructure = explore_speaker_substructure(
        array, candidate.labels
    )
    result["substructure_diagnostics"] = substructure.details
    result["provisional_outlier_count"] = len(flags)
    result["provisional_outliers"] = [{
        **flag,
        "window_start_ms": next(
            e["start_ms"] for e in events
            if e.get("vector_index") == flag["sample_index"]
        ),
    } for flag in flags]
    result["raw_cluster_count"] = len(set(candidate.labels))
    result["provisional_cluster_count"] = len(set(provisional))

    reference_ids = []
    original_labels = []
    provisional_labels = []
    per_person: dict[str, dict[int, int]] = {
        speaker: {} for speaker in result["pure_voiceprint_counts"]
    }
    for index, info in evaluations:
        ref_id = info["reference_speaker"]
        if ref_id is None:
            continue
        raw = int(candidate.labels[index])
        counter = per_person[ref_id]
        counter[raw] = counter.get(raw, 0) + 1
        reference_ids.append(ref_id)
        original_labels.append(raw)
        provisional_labels.append(int(provisional[index]))
    result["per_reference_speaker"] = [{
        "reference_speaker": speaker,
        "pure_window_embeddings": result["pure_voiceprint_counts"][speaker],
        "raw_cluster_contacts": [
            {"cluster": label, "windows": count}
            for label, count in sorted(
                per_person[speaker].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    } for speaker in sorted(per_person)]
    if len(reference_ids) >= 2:
        pure_vector_indices = [
            index for index, info in evaluations
            if info["reference_speaker"] is not None
        ]
        result["substructure_pure_window_scores"] = [
            {
                "variant": name,
                "all_window_cluster_count": len(set(map(int, assignments))),
                "pure_window_score": score_reference_partition(
                    reference_ids,
                    assignments[pure_vector_indices],
                ),
            }
            for name, assignments in substructure.variants.items()
        ]
        # Sequential hypotheses: first separate a coherent subgroup,
        # then inspect whether its smaller new group contains a lone
        # acoustically isolated voice. Reference labels are consulted
        # only in the final score, never to select either partition.
        # These candidates are NOT published speaker assignments.
        event_times = {
            event["vector_index"]: event["start_ms"]
            for event in events
            if event.get("vector_index") is not None
        }
        result["hierarchical_provisional_review"] = []
        for name, assignments in substructure.variants.items():
            if name == "original":
                continue
            revised, hypotheses = provisional_isolated_members(
                array, assignments
            )
            result["hierarchical_provisional_review"].append({
                "source_variant": name,
                "original_cluster_count": len(set(map(int, assignments))),
                "provisional_cluster_count": len(set(map(int, revised))),
                "unconfirmed_candidate_count": len(hypotheses),
                "unconfirmed_candidates": [
                    {**hypothesis,
                     "window_start_ms": event_times[hypothesis["sample_index"]]}
                    for hypothesis in hypotheses
                ],
                "pure_window_score": score_reference_partition(
                    reference_ids,
                    revised[pure_vector_indices],
                ),
                "warning": (
                    "Two-stage acoustic hypotheses only. A second split "
                    "might be noise, overlap, or microphone drift, NOT a "
                    "new human speaker. No live label is modified."
                ),
            })
        result["pure_window_sample_scores"] = {
            "spherical": score_reference_partition(
                reference_ids, np.asarray(original_labels)
            ),
            "provisional": score_reference_partition(
                reference_ids, np.asarray(provisional_labels)
            ),
            "warning": (
                "Only reference-pure windows are scored, but all successful "
                "windows (including mixed and unannotated ones) were clustered. "
                "Neither method used reference labels for clustering. "
                "These scores are not meeting DER."
            ),
        }
    result["available"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("benchmark/fixed195/ami_shadow10"),
    )
    parser.add_argument(
        "--meeting", choices=[m for m, _ in SELECTED], required=True
    )
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    private = Path("benchmark/fixed195").resolve()
    if (not root.is_relative_to(private)
            or not output.is_relative_to(root)
            or output.suffix.lower() != ".json"
            or output.exists()):
        parser.error("Expected a new private Fixed195 JSON output")
    baseline = score_case(root, args.meeting, args.replay)
    case = json.loads(
        (root / "cases" / (args.meeting + ".json")).read_text(encoding="utf-8")
    )
    audio_path = (root / case["audio"]).resolve()
    if not audio_path.is_relative_to(root):
        raise ValueError("Source escaped AMI root")
    samples = load_wav(audio_path)
    if len(samples) != 240 * SAMPLE_RATE:
        raise ValueError("Expected exactly 240s source audio")
    from core.config import Settings
    from inference.models import ModelManager
    from speech.diarization.speaker_backends import CamPlusEmbedding
    manager = ModelManager(Settings())
    try:
        vectors, events = extract_continuous_windows(
            samples, CamPlusEmbedding(manager)
        )
    finally:
        manager.close()
    report = audit_continuous_windows(
        vectors, events, case["reference_turns"]
    )
    report.update({
        "schema": "meeting-ami-continuous-camplus-2s-audit-v1",
        "meeting": args.meeting,
        "source_audio_sha256": baseline["source_audio_sha256"],
        "reference_source_sha256": baseline["reference_source_sha256"],
        "baseline_replay_sha256": baseline["replay_sha256"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

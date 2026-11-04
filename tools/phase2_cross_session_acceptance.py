"""Cross-session speaker identity calibration with disjoint evaluation.

Input is a private, human-labeled comparison manifest containing only
opaque session IDs, opaque speaker-group IDs and cosine scores. Embeddings,
names and transcription text are NEVER copied into aggregate output.
A development-set threshold may not be selected using holdout labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "meeting-phase2-cross-session-trials-v1"
OUTPUT_SCHEMA = "meeting-phase2-cross-session-acceptance-v1"


def validate_trials(payload: dict, *, min_positive: int = 20,
                    min_negative: int = 20) -> list[dict]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unknown cross-session identity trial format")
    rows = payload.get("trials")
    if not isinstance(rows, list) or len(rows) > 50000:
        raise ValueError("Invalid trial count")
    ids = set()
    unique_clip_pairs = set()
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed trial")
        id_ = row.get("id")
        src_a, src_b = row.get("session_a"), row.get("session_b")
        speaker_a, speaker_b = row.get("speaker_a"), row.get("speaker_b")
        clip_a, clip_b = row.get("clip_a"), row.get("clip_b")
        model = row.get("embedding_model")
        score = row.get("cosine")
        label = row.get("same_person")
        if (not isinstance(id_, str) or not id_ or id_ in ids
                or any(not isinstance(x, str) or not x or len(x) > 96
                       for x in (src_a, src_b, speaker_a, speaker_b,
                                 clip_a, clip_b, model))
                or src_a == src_b or clip_a == clip_b
                or type(label) is not bool
                or (speaker_a == speaker_b) != label
                or type(score) not in (float, int)
                or not math.isfinite(score) or not -1 <= score <= 1):
            raise ValueError("Invalid trial identity, cosine or cross-session provenance")
        if not row.get("human_labeled") is True:
            raise ValueError("Cross-session labels require independent human annotation")
        pair_key = (model, *sorted((clip_a, clip_b)))
        if pair_key in unique_clip_pairs:
            raise ValueError("Duplicate source clips would inflate pair counts")
        ids.add(id_)
        unique_clip_pairs.add(pair_key)
        samples.append({
            "id": id_, "session_a": src_a, "session_b": src_b,
            "speaker_a": speaker_a, "speaker_b": speaker_b,
            "clip_a": clip_a, "clip_b": clip_b,
            "embedding_model": model,
            "same_person": label, "cosine": float(score),
        })
    positives = sum(x["same_person"] for x in samples)
    negatives = len(samples) - positives
    if positives < min_positive or negatives < min_negative:
        raise ValueError("Insufficient genuine or impostor pairs")
    if len({s for x in samples for s in (x["session_a"], x["session_b"])}) < 3:
        raise ValueError("Insufficient independent recording sessions")
    if len({x["embedding_model"] for x in samples}) != 1:
        raise ValueError("Different embedding models cannot be calibrated together")
    return samples


def rates(trials: list[dict], threshold: float) -> dict:
    if not math.isfinite(threshold) or not -1.001 <= threshold <= 1.001:
        raise ValueError("Invalid threshold")
    positives = [x for x in trials if x["same_person"]]
    negatives = [x for x in trials if not x["same_person"]]
    false_reject = sum(x["cosine"] < threshold for x in positives)
    false_accept = sum(x["cosine"] >= threshold for x in negatives)
    return {
        "false_accept_rate": round(false_accept / len(negatives), 5),
        "false_reject_rate": round(false_reject / len(positives), 5),
        "false_accepts": false_accept,
        "false_rejects": false_reject,
        "genuine_pairs": len(positives),
        "impostor_pairs": len(negatives),
    }


def select_dev_threshold(rows: list[dict]) -> tuple[float, dict]:
    values = sorted({x["cosine"] for x in rows})
    trials = [-1.001, *values, 1.001]
    best = None
    for threshold in trials:
        result = rates(rows, threshold)
        error = abs(result["false_accept_rate"] - result["false_reject_rate"])
        ranking = (error, result["false_accept_rate"]
                   + result["false_reject_rate"], threshold)
        if best is None or ranking < best[0]:
            best = (ranking, threshold, result)
    assert best is not None
    return best[1], best[2]


def evaluate(dev_payload: dict, holdout_payload: dict, *,
             min_pairs: int = 20) -> dict:
    dev = validate_trials(dev_payload, min_positive=min_pairs,
                          min_negative=min_pairs)
    holdout = validate_trials(holdout_payload, min_positive=min_pairs,
                              min_negative=min_pairs)
    if dev[0]["embedding_model"] != holdout[0]["embedding_model"]:
        raise ValueError("Calibration and holdout embedding model differ")
    dev_sessions = {x[f"session_{key}"] for x in dev for key in ("a", "b")}
    holdout_sessions = {x[f"session_{key}"] for x in holdout for key in ("a", "b")}
    if dev_sessions & holdout_sessions:
        raise ValueError("Development and holdout sessions must be disjoint")
    dev_speakers = {x[key] for x in dev for key in ("speaker_a", "speaker_b")}
    holdout_speakers = {
        x[key] for x in holdout for key in ("speaker_a", "speaker_b")
    }
    if dev_speakers & holdout_speakers:
        raise ValueError("Development and holdout speaker identities must be disjoint")
    if {x["id"] for x in dev} & {x["id"] for x in holdout}:
        raise ValueError("Duplicate development/holdout trial IDs")
    threshold, dev_rates = select_dev_threshold(dev)
    holdout_rates = rates(holdout, threshold)
    passed = (
        holdout_rates["false_accept_rate"] <= 0.10
        and holdout_rates["false_reject_rate"] <= 0.20
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "embedding_model": dev[0]["embedding_model"],
        "dev_pairs": len(dev),
        "holdout_pairs": len(holdout),
        "dev_session_count": len(dev_sessions),
        "holdout_session_count": len(holdout_sessions),
        "threshold_selected_on_dev_only": threshold,
        "dev": dev_rates,
        "holdout": holdout_rates,
        "holdout_goal_met": passed,
        "may_auto_update_production_threshold": False,
        "note": (
            "This is only identity-verification calibration; it is not DER. "
            "Human-labeled cross-meeting pairs and disjoint sessions and "
            "speakers are required. Acceptance does not alter live threshold."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    root = Path("benchmark/fixed195").resolve()
    files = [options.dev.resolve(), options.holdout.resolve(),
             options.output.resolve()]
    if (any(not p.is_relative_to(root) for p in files)
            or files[-1].suffix != ".json" or files[-1].exists()):
        parser.error("Use private Fixed195 manifests and a NEW private JSON output")
    result = evaluate(
        json.loads(files[0].read_text(encoding="utf-8")),
        json.loads(files[1].read_text(encoding="utf-8")),
    )
    files[-1].parent.mkdir(parents=True, exist_ok=True)
    with files[-1].open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["holdout_goal_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

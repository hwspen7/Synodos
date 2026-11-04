"""Create independently labeled cross-session voiceprint trials.

Only manually corrected, explicitly human-verified enrollment vectors are
eligible. Automatic CAM++ or Nemotron assignments do not supply truth.
Read from the existing local/PostgreSQL SpeakerStore; never export embeddings.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.phase2_cross_session_acceptance import SCHEMA, validate_trials


def eligible_vectors(
    profiles: list[dict], *, model_id: str, meetings: set[str],
) -> list[dict]:
    seen_vectors = set()
    out = []
    for person in profiles:
        sid = str(person["id"])
        for row in person.get("vectors") or []:
            vector_id = str(row.get("id") or "")
            session = row.get("meeting_id")
            if (row.get("embedding_model", "camplus") != model_id
                    or row.get("human_verified") is not True
                    or row.get("verification_source") != "manual_profile_correction"
                    or row.get("short") is True
                    or type(session) is not str or session not in meetings
                    or not vector_id or vector_id in seen_vectors):
                continue
            try:
                embedding = np.asarray(row.get("emb"), dtype=np.float64).reshape(-1)
            except (ValueError, TypeError):
                continue
            if (not 32 <= embedding.size <= 4096
                    or not np.isfinite(embedding).all()):
                continue
            magnitude = float(np.linalg.norm(embedding))
            if not np.isfinite(magnitude) or magnitude <= 1e-8:
                continue
            seen_vectors.add(vector_id)
            out.append({
                "speaker": sid,
                "session": session,
                "clip": vector_id,
                "vector": embedding / magnitude,
            })
    return sorted(out, key=lambda x: (x["speaker"], x["session"], x["clip"]))


def build_trials(
    profiles: list[dict], *, model_id: str, meetings: set[str],
    limit_per_class: int = 5000,
) -> dict:
    if not 20 <= limit_per_class <= 10000:
        raise ValueError("Invalid pair cap")
    vectors = eligible_vectors(profiles, model_id=model_id, meetings=meetings)
    positive, negative = [], []
    for a, b in itertools.combinations(vectors, 2):
        if a["session"] == b["session"]:
            continue
        same = a["speaker"] == b["speaker"]
        entry = {
            "id": f"trial-{len(positive) + len(negative)}",
            "session_a": a["session"], "session_b": b["session"],
            "clip_a": a["clip"], "clip_b": b["clip"],
            "speaker_a": a["speaker"], "speaker_b": b["speaker"],
            "embedding_model": model_id,
            "human_labeled": True,
            "same_person": same,
            "cosine": round(float(np.dot(a["vector"], b["vector"])), 8),
        }
        if same and len(positive) < limit_per_class:
            positive.append(entry)
        elif not same and len(negative) < limit_per_class:
            negative.append(entry)
        if len(positive) >= limit_per_class and len(negative) >= limit_per_class:
            break
    # The final evaluator will enforce an adequate number of human-labeled
    # pairs and distinct sessions. Never duplicate or fabricate trials.
    report = {"schema": SCHEMA, "trials": positive + negative}
    validate_trials(report)
    return report


def _names(raw: str) -> set[str]:
    values = {s.strip() for s in raw.split(",") if s.strip()}
    if not values:
        raise ValueError("Explicit profile and meeting partitions are required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-profiles", required=True,
                        help="Comma-separated enrolled speaker IDs")
    parser.add_argument("--holdout-profiles", required=True)
    parser.add_argument("--dev-meetings", required=True,
                        help="Comma-separated meeting IDs")
    parser.add_argument("--holdout-meetings", required=True)
    parser.add_argument("--embedding-model", default="camplus")
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    args = parser.parse_args(argv)
    dev_people, test_people = _names(args.dev_profiles), _names(args.holdout_profiles)
    dev_sessions, test_sessions = _names(args.dev_meetings), _names(args.holdout_meetings)
    if dev_people & test_people or dev_sessions & test_sessions:
        parser.error("Dev and holdout speaker IDs and meeting IDs must be disjoint")
    base = Path("benchmark/fixed195").resolve()
    outputs = [args.dev_output.resolve(), args.holdout_output.resolve()]
    if (outputs[0] == outputs[1] or any(
        not out.is_relative_to(base) or out.suffix != ".json" or out.exists()
        for out in outputs
    )):
        parser.error("Use two NEW private Fixed195 JSON paths")
    from core.runtime import speaker_store
    dev_profiles = speaker_store.get_for_matching(sorted(dev_people))
    test_profiles = speaker_store.get_for_matching(sorted(test_people))
    if len(dev_profiles) != len(dev_people) or len(test_profiles) != len(test_people):
        raise ValueError("One or more enrolled speaker profiles do not exist")
    reports = [
        build_trials(dev_profiles, model_id=args.embedding_model,
                     meetings=dev_sessions),
        build_trials(test_profiles, model_id=args.embedding_model,
                     meetings=test_sessions),
    ]
    for path, result in zip(outputs, reports):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    print(json.dumps({
        "schema": "meeting-phase2-trial-building-receipt-v1",
        "dev_pairs": len(reports[0]["trials"]),
        "holdout_pairs": len(reports[1]["trials"]),
        "dev_output": str(outputs[0]),
        "holdout_output": str(outputs[1]),
        "no_embeddings_exported": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

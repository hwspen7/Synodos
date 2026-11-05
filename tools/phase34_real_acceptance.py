"""Local held-out real-audio RAG evaluation (only the meeting_test database).

No synthetic case can satisfy this gate. Human-reviewed source audio and exact
meeting/segment timestamps must be supplied independently. This command never
edits meeting transcripts, deletes meetings or starts the live ASR backend.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path

from tools.phase34_release_contract import (
    load_json, real_summary, score_answerable, score_negative,
    sha256_file, validate_manifest, write_new,
)


def source_gate(case, segments):
    """Fail if expected human-audio fact cannot be found in saved ASR text."""
    match = case["expected_text"].casefold()
    gold = set(case["gold_segment_ids"])
    start, end = case["gold_interval_ms"]
    return any(
        s.get("id") in gold and
        match in str(s.get("text") or "").casefold() and
        s.get("start_ms", -1) < end and s.get("end_ms", -1) > start
        for s in segments
    )


def negative_witness_gate(case, meeting_state):
    """Ground the negative type in saved, human-reviewed source witnesses."""
    if case["kind"] == "cross_scope":
        foreign = case["foreign_meeting_id"]
        fact = case["foreign_expected_text"].casefold()
        foreign_segments = meeting_state[foreign][1]
        target_segments = meeting_state[case["meeting_id"]][1]
        return (any(fact in s["text"].casefold() for s in foreign_segments) and
                not any(fact in s["text"].casefold() for s in target_segments))
    if case["kind"] == "contradiction":
        ids = set(case["contradiction_segment_ids"])
        return ids.issubset({s["id"] for s in meeting_state[case["meeting_id"]][1]})
    return True


def evaluate(args):
    # The existing synthetic runner checks DB name, active realtime sessions,
    # PostgreSQL configuration and local Ollama availability.
    from tools.phase34_full_acceptance import guard_environment
    guard_environment()
    from intelligence.config import load_config
    from intelligence.model_gateway import ModelGateway
    from intelligence import retrieval, store
    from intelligence.worker import process

    manifest = load_json(args.manifest)
    positives, negatives = validate_manifest(manifest)
    cfg = load_config()
    if not cfg.enabled or cfg.provider != "ollama":
        raise RuntimeError("Use a real local Ollama model, not a stub")
    if cfg.enable_embeddings != (args.mode == "hybrid"):
        raise RuntimeError("MEETING_AI_EMBEDDINGS does not match requested mode")
    gateway = ModelGateway(cfg)
    if args.mode == "hybrid" and not gateway.embed("real benchmark preflight"):
        raise RuntimeError("The configured embedding model is unavailable")
    mids = [x["meeting_id"] for x in manifest["meetings"]]
    meeting_state = {}
    for audio in manifest["meetings"]:
        audio_file = Path(audio["audio_file"])
        if not audio_file.is_file():
            raise RuntimeError("Real source audio missing for meeting " + audio["meeting_id"])
        if sha256_file(audio_file).lower() != audio["audio_sha256"].lower():
            raise RuntimeError("Real source audio fingerprint mismatch: " + audio["meeting_id"])
    for mid in mids:
        revision, segments = store.meeting_segments(mid)
        if not segments:
            raise RuntimeError("Reviewed meeting has no saved segments: " + mid)
        meeting_state[mid] = (revision, segments)
        if args.reindex:
            # Rebuild only search index; source transcript is never modified.
            out = process({"id": str(uuid.uuid4()), "kind": "index_meeting",
                           "meeting_id": mid, "payload": {}}, gateway)
            if out.get("source_revision") != revision:
                raise RuntimeError("Index revision mismatch: " + mid)
    indexed_count = dict.fromkeys(mids, 0)
    for hit in store.iter_knowledge_candidates(meeting_ids=mids):
        mid = hit.get("source_id")
        if mid not in indexed_count or hit.get("revision") != meeting_state[mid][0]:
            raise RuntimeError("Invalid current-revision knowledge index")
        indexed_count[mid] += 1
    if any(count == 0 for count in indexed_count.values()):
        raise RuntimeError("One or more real reviewed meetings lack a current index")
    results, timings = [], []
    for case in manifest["cases"]:
        mid = case["meeting_id"]
        start = time.monotonic()
        case_key = hashlib.sha256(case["id"].encode("utf-8")).hexdigest()[:16]
        try:
            hits = retrieval.search(case["question"], gateway, limit=8, meeting_ids=[mid])
            out = process({"id": str(uuid.uuid4()), "kind": "ask_evidence",
                           "payload": {"question": case["question"], "meeting_ids": [mid]}}, gateway)
            if case["kind"] == "answerable":
                current = store.evidence_revisions_current(out.get("sources") or [])
                metrics = score_answerable(case, hits, out, current)
                metrics["gold_present_in_saved_asr"] = source_gate(case, meeting_state[mid][1])
                # Audio-only words missing from ASR are never counted as correct.
                if not metrics["gold_present_in_saved_asr"]:
                    metrics["answer_correct"] = False
                    metrics["citation_attribution_correct"] = False
                passed = metrics["answer_correct"]
            else:
                metrics = score_negative(case, out, hits)
                metrics["human_negative_witness_present"] = negative_witness_gate(case, meeting_state)
                if not metrics["human_negative_witness_present"]:
                    metrics["abstention_correct"] = False
                passed = metrics["abstention_correct"]
            note = ""
        except Exception as exc:
            metrics, passed = {}, False
            note = type(exc).__name__ + ": " + str(exc)[:160]
        seconds = round(time.monotonic() - start, 3)
        timings.append(seconds)
        results.append({
            "id_hash": case_key, "meeting_id_hash": hashlib.sha256(mid.encode()).hexdigest()[:16],
            "kind": case["kind"], "language": case["language"],
            "passed": bool(passed), "seconds": seconds, "note": note,
            **metrics,
        })
        print(f"REAL {case['kind']:<12} {'PASS' if passed else 'FAIL'} {seconds:7.3f}s {case_key}")
    report = real_summary(results, mode=args.mode, timings=timings,
                          text_model=cfg.text_model,
                          embed_model=cfg.embed_model if cfg.enable_embeddings else None)
    report.update({
        "source_manifest_sha256": sha256_file(args.manifest),
        "human_reviewed_meeting_count": len(mids),
        "source_commit": os.environ.get("MEETING_SOURCE_COMMIT", ""),
        "index_action": "reindexed_current_test_records" if args.reindex else "read_only_existing_index",
        "audio_diarization_gate_evaluated": False,
        "automatic_external_actions": False,
        "private_questions_in_report": False,
    })
    write_new(args.output, report)
    print("REAL AUDIO PHASE 3/4:", "PASS" if report["pass"] else "NOT_ACCEPTED")
    print("REPORT:", args.output)
    return 0 if report["pass"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("lexical", "hybrid"), required=True)
    p.add_argument("--reindex", action="store_true", help="Rebuild knowledge index in meeting_test")
    a = p.parse_args(argv)
    try:
        if a.output.exists():
            raise ValueError("Output already exists; immutable acceptance reports only")
        return evaluate(a)
    except Exception as exc:
        print("REAL AUDIO PHASE 3/4 NOT_ACCEPTED:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

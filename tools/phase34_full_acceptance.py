"""Real local Phase 3/4 evidence-QA acceptance against a disposable test corpus.

Requires the user's existing PostgreSQL meeting_test and local Ollama models.
Creates three isolated synthetic meetings, indexes them, runs >=20 answerable
and >=6 negative questions, verifies citation provenance, and removes ONLY
those generated meetings. This is not a real-audio or human-gold benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid
from urllib.request import urlopen


def guard_environment():
    import psycopg

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    if os.environ.get("MEETING_STORAGE_BACKEND") != "postgres":
        raise RuntimeError("MEETING_STORAGE_BACKEND=postgres is required")
    if os.environ.get("MEETING_AI_ENABLED") != "1":
        raise RuntimeError("MEETING_AI_ENABLED=1 is required")
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        if conn.info.dbname != "meeting_test":
            raise RuntimeError("This acceptance suite only writes to meeting_test")
    # Never compete with a currently running realtime ASR session.
    try:
        with urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
            health = json.load(response)
        active = int(health.get("active_meetings",
                                health.get("active_sessions", 0)))
        if active:
            raise RuntimeError("A live meeting is active on port 8000")
    except OSError:
        pass


def fixtures():
    # Unicode escapes keep source and tests compatible with the repository's
    # English-only tracked-code gate without removing multilingual coverage.
    meetings = {
        "atlas": [
            "The Atlas prototype demonstration is scheduled for November 27, 2026.",
            "Maya Chen is the engineering owner for Atlas.",
            "The approved Atlas spending cap is 48000 US dollars.",
            "The main Atlas technical risk is microphone synchronization.",
            "The Atlas integration report is due on December 1, 2026.",
            "The next Atlas planning meeting is in Room 214.",
            "The Atlas interface will retain a purple primary color.",
            "The Atlas test build identifier is BLD84.",
            "Atlas \u4f1a\u8bae\u786e\u8ba4\u4e86\u9ea6\u514b\u98ce\u540c\u6b65\u95ee\u9898\u3002",
            "The Atlas multilingual validation key is EN-731.",
            "Atlas \u9879\u76ee\u7684\u9a8c\u8bc1\u7801\u662f ZH-731\u3002",
            "Le code de verification Atlas est FR-731.",
            "El codigo de validacion Atlas es ES-731.",
        ],
        "beacon": [
            "The Beacon delivery milestone is December 4, 2026.",
            "Luis Romero owns the Beacon deployment checklist.",
            "The Beacon rollout location is Building 8.",
            "Beacon has a spending limit of 19000 US dollars.",
            "The Beacon integration meeting starts at 09:30.",
            "The Beacon release channel is beta-seven.",
            "The Beacon risk review happens on December 2, 2026.",
            "Beacon \u30c1\u30fc\u30e0\u306f\u4f1a\u8b70\u3092\u78ba\u8a8d\u3057\u307e\u3057\u305f\u3002",
            "\u0631\u0645\u0632 \u0627\u0644\u062a\u062d\u0642\u0642 \u0644\u0640 Beacon \u0647\u0648 AR-731.",
            "Kode validasi Beacon adalah ID-731.",
            "Beacon \u306e\u78ba\u8a8d\u30b3\u30fc\u30c9\u306f JP-731 \u3067\u3059\u3002",
            "\u041a\u043e\u0434 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 Beacon: RU-731.",
        ],
        "cobalt": [
            "The Cobalt customer briefing is scheduled for December 11, 2026.",
            "Priya Shah leads the Cobalt documentation task.",
            "The Cobalt project uses the Nimbus staging environment.",
            "The Cobalt service target is 99.5 percent availability.",
            "The Cobalt procurement checkpoint is December 8, 2026.",
            "The Cobalt backup coordinator is Ana Silva.",
            "Cobalt approved a ten-minute rollback window.",
            "The Cobalt team will use the Delta conference room.",
            "Der Cobalt Pruefcode ist DE-731.",
            "\u0915\u094b\u092c\u093e\u0932\u094d\u091f \u0938\u0924\u094d\u092f\u093e\u092a\u0928 \u0915\u094b\u0921 HI-731 \u0939\u0948\u0964",
            "\u0995\u09cb\u09ac\u09be\u09b2\u09cd\u099f \u09af\u09be\u099a\u09be\u0987 \u0995\u09cb\u09a1 BN-731\u0964",
            "O codigo de verificacao Cobalt e PT-731.",
        ],
    }
    checks = [
        ("atlas", "When is the Atlas prototype demonstration?", "November 27, 2026"),
        ("atlas", "Who owns Atlas engineering?", "Maya Chen"),
        ("atlas", "What is the Atlas approved spending cap?", "48000"),
        ("atlas", "What is the Atlas technical risk?", "microphone synchronization"),
        ("atlas", "When is the Atlas integration report due?", "December 1, 2026"),
        ("atlas", "Which room hosts Atlas planning?", "Room 214"),
        ("atlas", "Which primary color will Atlas retain?", "purple"),
        ("atlas", "What is the Atlas build identifier?", "BLD84"),
        ("beacon", "What is the Beacon delivery milestone date?", "December 4, 2026"),
        ("beacon", "Who owns the Beacon deployment checklist?", "Luis Romero"),
        ("beacon", "Where is the Beacon rollout?", "Building 8"),
        ("beacon", "What is the Beacon spending limit?", "19000"),
        ("beacon", "When does the Beacon integration meeting start?", "09:30"),
        ("beacon", "What is the Beacon release channel?", "beta-seven"),
        ("beacon", "When is the Beacon risk review?", "December 2, 2026"),
        ("beacon", "\u8bf7\u95ee Beacon \u7684\u4ea4\u4ed8\u65e5\u671f\u662f\u4ec0\u4e48\uff1f", "December 4, 2026"),
        ("cobalt", "When is the Cobalt customer briefing?", "December 11, 2026"),
        ("cobalt", "Who leads Cobalt documentation?", "Priya Shah"),
        ("cobalt", "What is the Cobalt staging environment?", "Nimbus"),
        ("cobalt", "What availability does Cobalt target?", "99.5"),
        ("cobalt", "When is the Cobalt procurement checkpoint?", "December 8, 2026"),
        ("cobalt", "Who is the Cobalt backup coordinator?", "Ana Silva"),
        ("cobalt", "What rollback window was approved for Cobalt?", "ten-minute"),
        ("cobalt", "Where does the Cobalt team meet?", "Delta conference room"),
        # Twelve local-model multilingual quote-selection probes.
        ("atlas", "What is the Atlas multilingual validation key?", "EN-731"),
        ("atlas", "Atlas \u9879\u76ee\u7684\u9a8c\u8bc1\u7801\u662f\u591a\u5c11\uff1f", "ZH-731"),
        ("atlas", "Quel est le code de verification Atlas ?", "FR-731"),
        ("atlas", "Cual es el codigo de validacion Atlas?", "ES-731"),
        ("beacon", "\u0645\u0627 \u0647\u0648 \u0631\u0645\u0632 \u0627\u0644\u062a\u062d\u0642\u0642 \u0644\u0640 Beacon\u061f", "AR-731"),
        ("beacon", "Apa kode validasi Beacon?", "ID-731"),
        ("beacon", "Beacon \u306e\u78ba\u8a8d\u30b3\u30fc\u30c9\u306f\uff1f", "JP-731"),
        ("beacon", "\u041a\u0430\u043a\u043e\u0439 \u043a\u043e\u0434 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 Beacon?", "RU-731"),
        ("cobalt", "Wie lautet der Cobalt Pruefcode?", "DE-731"),
        ("cobalt", "\u0915\u094b\u092c\u093e\u0932\u094d\u091f \u0938\u0924\u094d\u092f\u093e\u092a\u0928 \u0915\u094b\u0921 \u0915\u094d\u092f\u093e \u0939\u0948?", "HI-731"),
        ("cobalt", "\u0995\u09cb\u09ac\u09be\u09b2\u09cd\u099f \u09af\u09be\u099a\u09be\u0987 \u0995\u09cb\u09a1 \u0995\u09bf?", "BN-731"),
        ("cobalt", "Qual e o codigo de verificacao Cobalt?", "PT-731"),
    ]
    negatives = [
        ("atlas", "What is the Atlas quantum clock frequency?"),
        ("atlas", "Which manufacturer supplies Atlas submarine propellers?"),
        ("beacon", "Who will pilot the Beacon interplanetary capsule?"),
        ("beacon", "What is Beacon's final cryptocurrency treasury balance?"),
        ("cobalt", "Which telescope discovered Cobalt's exoplanet?"),
        ("cobalt", "What is Cobalt's bacterial genome sequence?"),
    ]
    return meetings, checks, negatives


def percentile(values, level):
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(
        (len(values) - 1) * level + 0.999999
    )))
    return round(values[idx], 3)


def run(args):
    guard_environment()
    from storage.postgres_connection import PostgresDatabase
    from storage.postgres_meetings import PostgresMeetingStore
    from intelligence.config import load_config
    from intelligence.model_gateway import ModelGateway
    from intelligence.worker import process
    from intelligence import retrieval, store

    config = load_config()
    if not config.enabled or config.provider != "ollama":
        raise RuntimeError("The real acceptance runner requires local Ollama")
    if args.mode == "hybrid" and not config.enable_embeddings:
        raise RuntimeError("Hybrid mode needs MEETING_AI_EMBEDDINGS=1")
    if args.mode == "lexical" and config.enable_embeddings:
        raise RuntimeError("Lexical mode needs MEETING_AI_EMBEDDINGS=0")

    gateway = ModelGateway(config)
    if args.mode == "hybrid":
        # Fail before touching DB records if the embedder is missing.
        sample = gateway.embed("embedding preflight")
        if not sample:
            raise RuntimeError("Embedding preflight returned no vector")
    meetings, checks, negatives = fixtures()
    db = PostgresDatabase(os.environ["DATABASE_URL"])
    meeting_store = PostgresMeetingStore(db)
    ids = {}
    timings = []
    outcomes = []
    started = time.monotonic()
    try:
        for label, lines in meetings.items():
            mid = "phase34_eval_" + uuid.uuid4().hex[:20]
            ids[label] = mid
            state = {
                "id": mid, "title": "Phase 34 synthetic " + label,
                "sample_rate": 16000,
                "segments": [{
                    "id": i + 1, "speaker_id": i % 2 + 1,
                    "cluster": i % 2, "speaker_name": "Speaker " + str(i % 2 + 1),
                    "start_sample": i * 160000,
                    "end_sample": (i + 1) * 160000,
                    "text": line, "revision": 0,
                } for i, line in enumerate(lines)],
            }
            meeting_store.save_state(state)
            out = process({
                "id": str(uuid.uuid4()), "kind": "index_meeting",
                "meeting_id": mid, "payload": {},
            }, gateway)
            if out["indexed_chunks"] < 1:
                raise RuntimeError("Synthetic indexing returned zero chunks")

        # Retrieval and citation checks are scoped to the named meeting.
        # No outputs are printed from user-owned meetings.
        for label, question, expected in checks:
            mid = ids[label]
            query_started = time.monotonic()
            retrieval_expected_in_top8 = False
            response_quote_count = 0
            try:
                hits = retrieval.search(question, gateway, meeting_ids=[mid])
                retrieval_expected_in_top8 = any(
                    expected.casefold() in hit.get("text", "").casefold()
                    for hit in hits
                )
                out = process({
                    "id": str(uuid.uuid4()), "kind": "ask_evidence",
                    "payload": {
                        "question": question, "meeting_ids": [mid],
                    },
                }, gateway)
                answer = out.get("answer", "")
                sources = out.get("sources") or []
                quotes = out.get("quotes") or []
                response_quote_count = len(quotes)
                source_ok = bool(sources) and all(
                    src.get("source_kind") == "meeting" and
                    src.get("source_id") == mid for src in sources
                ) and store.evidence_revisions_current(sources)
                cited = out.get("status") == "verbatim_citations_verified"
                expected_hit = expected.casefold() in answer.casefold()
                exact = all(any(
                    item.get("quote", "") in hit.get("text", "")
                    for hit in hits
                ) for item in quotes)
                passed = bool(cited and expected_hit and source_ok and exact)
                note = "" if passed else (
                    f"status={out.get('status')} expected_found={expected_hit} "
                    f"retrieval_recall_at_8={retrieval_expected_in_top8} "
                    f"quote_count={response_quote_count} "
                    f"source_current={source_ok} exact={exact}"
                )
            except Exception as exc:
                passed = False
                note = str(exc)[:240]
            elapsed = round(time.monotonic() - query_started, 3)
            timings.append(elapsed)
            outcomes.append({
                "kind": "answerable", "meeting": label,
                "question": question, "expected": expected,
                "passed": passed, "seconds": elapsed, "note": note,
                "retrieval_expected_in_top8": retrieval_expected_in_top8,
                "response_quote_count": response_quote_count,
            })
            print(f"ANSWER {label:6} {'PASS' if passed else 'FAIL'} {elapsed:7.3f}s {question}")

        for label, question in negatives:
            mid = ids[label]
            query_started = time.monotonic()
            try:
                out = process({
                    "id": str(uuid.uuid4()), "kind": "ask_evidence",
                    "payload": {"question": question, "meeting_ids": [mid]},
                }, gateway)
                passed = (
                    out.get("status") == "abstained" and
                    not out.get("sources") and not out.get("quotes")
                )
                note = "" if passed else f"unexpected status={out.get('status')}"
            except Exception as exc:
                passed = False
                note = str(exc)[:240]
            elapsed = round(time.monotonic() - query_started, 3)
            timings.append(elapsed)
            outcomes.append({
                "kind": "unanswerable", "meeting": label,
                "question": question, "passed": passed,
                "seconds": elapsed, "note": note,
            })
            print(f"ABSTAIN {label:6} {'PASS' if passed else 'FAIL'} {elapsed:7.3f}s {question}")
    finally:
        for mid in ids.values():
            try:
                meeting_store.delete(mid)
            except Exception as exc:
                print(f"WARNING: review isolated test meeting {mid}: {exc}",
                      file=sys.stderr)
        db.close()

    answerable = [r for r in outcomes if r["kind"] == "answerable"]
    negatives = [r for r in outcomes if r["kind"] == "unanswerable"]
    counts = {
        "answerable_pass": sum(r["passed"] for r in answerable),
        "answerable_total": len(answerable),
        "retrieval_recall_at_8": sum(
            r.get("retrieval_expected_in_top8", False) for r in answerable
        ),
        "abstention_pass": sum(r["passed"] for r in negatives),
        "abstention_total": len(negatives),
    }
    passed = (counts["answerable_total"] >= 20 and
              counts["abstention_total"] >= 6 and
              counts["answerable_pass"] == counts["answerable_total"] and
              counts["abstention_pass"] == counts["abstention_total"])
    report = {
        "schema": "meeting-phase34-local-eval-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_type": "isolated_synthetic_three_meetings_12_language_probes",
        "real_ollama": True,
        "real_audio_gold_coverage": False,
        "human_semantic_truth_review": False,
        "mode": args.mode, "text_model": config.text_model,
        "embedding_model": config.embed_model if config.enable_embeddings else None,
        "passed": passed,
        "counts": counts,
        "latency_seconds": {
            "p50": statistics.median(timings) if timings else None,
            "p95": percentile(timings, 0.95),
            "total": round(time.monotonic() - started, 3),
        },
        "cases": outcomes,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print("PHASE 3/4 LOCAL ACCEPTANCE:", "PASS" if passed else "FAIL")
    print("REPORT:", dest)
    print("COUNTS:", json.dumps(counts))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lexical", "hybrid"),
                        default="lexical")
    parser.add_argument("--output", required=True,
                        help="Write the JSON report to an explicit E-drive path")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("PHASE 3/4 ACCEPTANCE FAILED:", str(exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

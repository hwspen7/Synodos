"""Pure, strict Meeting knowledge release contracts. No production mutations."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SYNTHETIC_SCHEMA = "meeting-knowledge-synthetic-evaluation"
# Accept frozen historical local reports without emitting their old format.
LEGACY_SYNTHETIC_SCHEMA = "meeting-phase34-local-eval-v1"
REAL_SCHEMA = "meeting-knowledge-real-audio-qa"
FREEZE_SCHEMA = "meeting-knowledge-baseline-freeze"
RELEASE_SCHEMA = "meeting-knowledge-release-decision"
SUPPORTED_ASR_LANGUAGES = frozenset(("en", "zh", "fr", "es", "ar", "id", "ja", "ru", "de", "hi"))
MEETING_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def load_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def sha256_file(path):
    hash_ = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(part)
    return hash_.hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def strict_synthetic(report, mode):
    if report.get("schema") not in (SYNTHETIC_SCHEMA, LEGACY_SYNTHETIC_SCHEMA) or report.get("mode") != mode:
        raise ValueError("Incorrect synthetic report schema or mode")
    if report.get("passed") is not True or report.get("real_ollama") is not True:
        raise ValueError("Synthetic local Ollama acceptance not passed")
    if report.get("real_audio_gold_coverage") is not False:
        raise ValueError("Synthetic report has unexpected real-audio claim")
    if report.get("human_semantic_truth_review") is not False:
        raise ValueError("Synthetic report has unexpected human-verification claim")
    c = report.get("counts") or {}
    if (type(c.get("answerable_pass")) is not int or
            type(c.get("answerable_total")) is not int or
            type(c.get("retrieval_recall_at_8")) is not int or
            type(c.get("abstention_pass")) is not int or
            type(c.get("abstention_total")) is not int or
            (c["answerable_pass"], c["answerable_total"],
             c["retrieval_recall_at_8"], c["abstention_pass"],
             c["abstention_total"]) != (36, 36, 36, 6, 6)):
        raise ValueError("Synthetic report must contain exact 36/36 + 6/6 + recall 36")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 42:
        raise ValueError("Synthetic report must contain 42 outcomes")
    answers = [x for x in cases if isinstance(x, dict) and x.get("kind") == "answerable"]
    negatives = [x for x in cases if isinstance(x, dict) and x.get("kind") == "unanswerable"]
    if len(answers) != 36 or len(negatives) != 6 or not all(
        x.get("passed") is True for x in cases if isinstance(x, dict)
    ) or len(answers) + len(negatives) != 42:
        raise ValueError("Incomplete or failed individual synthetic cases")
    if not all(x.get("retrieval_expected_in_top8") is True for x in answers):
        raise ValueError("Synthetic recall is inconsistent with per-case evidence")
    if not isinstance(report.get("text_model"), str) or not report["text_model"]:
        raise ValueError("Missing exact model name")
    embed = report.get("embedding_model")
    if mode == "hybrid" and (not isinstance(embed, str) or not embed):
        raise ValueError("Hybrid report does not identify its embedder")
    if mode == "lexical" and embed is not None:
        raise ValueError("Lexical report unexpectedly used embeddings")
    return True


def validate_manifest(data):
    """Require real audio annotation, not constructed text fixtures."""
    if data.get("schema") != "meeting-audio-review-gold":
        raise ValueError("Unknown real-audio gold manifest schema")
    if data.get("corpus_type") != "real_recorded_meetings":
        raise ValueError("Only real recorded meetings are admissible")
    if data.get("gold_source") != "independent_human_audio_review":
        raise ValueError("Gold labels must be independently verified from audio")
    if not isinstance(data.get("reviewer_id"), str) or not data["reviewer_id"].strip():
        raise ValueError("An actual reviewer identifier is required")
    meetings = data.get("meetings")
    if not isinstance(meetings, list) or not 3 <= len(meetings) <= 20:
        raise ValueError("At least three real recordings required")
    meeting_map = {}
    for m in meetings:
        if not isinstance(m, dict) or not MEETING_ID.fullmatch(str(m.get("meeting_id", ""))):
            raise ValueError("Invalid real meeting identifier")
        mid = m["meeting_id"]
        audio_hash = m.get("audio_sha256")
        if mid in meeting_map or not isinstance(audio_hash, str) or not SHA256.fullmatch(audio_hash):
            raise ValueError("Duplicate meeting or missing audio SHA256")
        if not isinstance(m.get("audio_file"), str) or not m["audio_file"].strip():
            raise ValueError("Real recording path is required for SHA256 verification")
        if m.get("audio_reviewed") is not True or m.get("synthetic_audio") is not False:
            raise ValueError("All source audio must be real and human reviewed")
        meeting_map[mid] = m
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Missing real-audio gold cases")
    seen = set()
    positives = []
    negatives = []
    for c in cases:
        if not isinstance(c, dict) or not CASE_ID.fullmatch(str(c.get("id", ""))):
            raise ValueError("Invalid case ID")
        if c["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(c["id"])
        if c.get("meeting_id") not in meeting_map:
            raise ValueError("Case references unknown meeting")
        if c.get("language") not in SUPPORTED_ASR_LANGUAGES:
            raise ValueError("Unsupported language in current ten-language audio gate")
        if (not isinstance(c.get("question"), str) or
                not 2 <= len(c["question"].strip()) <= 2000 or
                c.get("verified_from_audio") is not True):
            raise ValueError("Question must be reviewed against recorded audio")
        if c.get("kind") == "answerable":
            if (not isinstance(c.get("expected_text"), str) or
                    not 2 <= len(c["expected_text"].strip()) <= 600):
                raise ValueError("Answerable cases need independently reviewed expected text")
            ids = c.get("gold_segment_ids")
            if (not isinstance(ids, list) or not ids or
                    any(type(x) is not int or x < 0 for x in ids)):
                raise ValueError("Answerable cases need gold segment IDs")
            span = c.get("gold_interval_ms")
            if (not isinstance(span, list) or len(span) != 2 or
                    any(type(v) is not int or v < 0 for v in span) or span[0] >= span[1]):
                raise ValueError("Answerable cases need real audio interval")
            positives.append(c)
        elif c.get("kind") in ("unanswerable", "cross_scope", "contradiction"):
            if c.get("expected_text") or c.get("gold_segment_ids"):
                raise ValueError("Negative cases must not contain target answer")
            if c["kind"] == "cross_scope":
                foreign = c.get("foreign_meeting_id")
                if foreign not in meeting_map or foreign == c["meeting_id"]:
                    raise ValueError("Cross-scope case requires another reviewed meeting")
                if (not isinstance(c.get("foreign_expected_text"), str) or
                        len(c["foreign_expected_text"].strip()) < 2):
                    raise ValueError("Cross-scope must name the fact found only in the foreign meeting")
            if c["kind"] == "contradiction":
                witnesses = c.get("contradiction_segment_ids")
                if (not isinstance(witnesses, list) or len(set(witnesses)) < 2 or
                        any(type(x) is not int or x < 0 for x in witnesses)):
                    raise ValueError("Contradiction requires two audio-reviewed conflicting segments")
            negatives.append(c)
        else:
            raise ValueError("Unsupported gold case kind")
    if len(positives) < 20 or len(negatives) < 6:
        raise ValueError("At least 20 real positives and six real negatives required")
    if len({c["meeting_id"] for c in positives}) < 3:
        raise ValueError("Positive cases must span three recorded meetings")
    if not any(c["kind"] == "cross_scope" for c in negatives):
        raise ValueError("At least one cross-meeting isolation probe required")
    if not any(c["kind"] == "contradiction" for c in negatives):
        raise ValueError("At least one contradictory-evidence probe required")
    return positives, negatives


def _segments_overlap(gold_ids, evidence):
    ids = evidence.get("segment_ids") if isinstance(evidence, dict) else []
    return isinstance(ids, list) and bool(set(ids) & set(gold_ids))


def _time_overlap(span, evidence):
    if not isinstance(evidence, dict):
        return False
    start, end = evidence.get("start_ms"), evidence.get("end_ms")
    return (type(start) is int and type(end) is int and
            end > span[0] and start < span[1])


def score_answerable(case, hits, out, current):
    """Compute independent retrieval, attribution, extractive-answer checks."""
    expected = case["expected_text"].casefold()
    gold = set(case["gold_segment_ids"])
    span = case["gold_interval_ms"]
    def match(hit):
        return (hit.get("source_kind") == "meeting" and
                hit.get("source_id") == case["meeting_id"] and
                expected in str(hit.get("text") or "").casefold() and
                _segments_overlap(gold, hit.get("evidence")) and
                _time_overlap(span, hit.get("evidence")))
    recall = any(match(h) for h in hits[:8])
    sources = out.get("sources") or []
    quotes = out.get("quotes") or []
    hits_scoped = all(
        h.get("source_kind") == "meeting" and h.get("source_id") == case["meeting_id"]
        for h in hits
    )
    scoped = bool(sources) and all(
        s.get("source_kind") == "meeting" and
        s.get("source_id") == case["meeting_id"] for s in sources
    )
    cited_gold = any(match(s) for s in sources)
    quote_exact = bool(quotes) and all(
        type(q.get("source_number")) is int and
        1 <= q["source_number"] <= len(hits[:8]) and
        isinstance(q.get("quote"), str) and
        q["quote"] in str(hits[q["source_number"] - 1].get("text") or "")
        for q in quotes if isinstance(q, dict)
    ) and all(isinstance(q, dict) for q in quotes)
    truth_found = any(
        expected in q["quote"].casefold() and
        match(hits[q["source_number"] - 1])
        for q in quotes if isinstance(q, dict) and
        type(q.get("source_number")) is int and
        1 <= q["source_number"] <= len(hits[:8]) and
        isinstance(q.get("quote"), str)
    )
    selected_hits = [hits[q["source_number"] - 1] for q in quotes
                     if isinstance(q, dict) and type(q.get("source_number")) is int
                     and 1 <= q["source_number"] <= len(hits[:8])]
    provenance_matches = bool(selected_hits) and len(sources) == len({
        id(h) for h in selected_hits
    }) and all(any(
        s.get("source_kind") == h.get("source_kind") and
        s.get("source_id") == h.get("source_id") and
        s.get("revision") == h.get("revision") and
        s.get("text") == h.get("text") and
        s.get("evidence") == h.get("evidence")
        for s in sources
    ) for h in selected_hits)
    attribution = (out.get("status") == "verbatim_citations_verified" and
                   scoped and hits_scoped and current is True and
                   cited_gold and quote_exact and provenance_matches)
    return {
        "retrieval_recall_at_8": bool(recall),
        "citation_attribution_correct": bool(attribution),
        "answer_correct": bool(attribution and truth_found),
        "cross_scope_leak": bool(not hits_scoped or (sources and not scoped)),
    }


def score_negative(case, out, hits=None):
    sources = out.get("sources") or []
    quotes = out.get("quotes") or []
    return {
        "abstention_correct": (out.get("status") == "abstained" and
                               not sources and not quotes),
        "cross_scope_leak": any(
            s.get("source_id") != case["meeting_id"]
            for s in (sources + (hits or [])) if isinstance(s, dict)
        ),
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def real_summary(cases, *, mode, timings, text_model, embed_model):
    pos = [c for c in cases if c["kind"] == "answerable"]
    neg = [c for c in cases if c["kind"] != "answerable"]
    n_pos, n_neg = len(pos), len(neg)
    recall = sum(bool(c.get("retrieval_recall_at_8")) for c in pos)
    correct = sum(bool(c.get("answer_correct")) for c in pos)
    citation = sum(bool(c.get("citation_attribution_correct")) for c in pos)
    abstain = sum(bool(c.get("abstention_correct")) for c in neg)
    leaks = sum(bool(c.get("cross_scope_leak")) for c in cases)
    p95 = percentile(timings, 0.95)
    thresholds = {
        "real_gold_at_least_20": n_pos >= 20,
        "real_negatives_at_least_6": n_neg >= 6,
        "retrieval_recall_at_8_at_least_0_90": n_pos > 0 and recall / n_pos >= .90,
        "answer_correct_at_least_0_85": n_pos > 0 and correct / n_pos >= .85,
        "citation_attribution_at_least_0_95": n_pos > 0 and citation / n_pos >= .95,
        "abstention_at_least_0_95": n_neg > 0 and abstain / n_neg >= .95,
        "zero_cross_scope_leaks": leaks == 0,
        "qa_latency_p95_at_most_15s": p95 is not None and p95 <= 15.0,
    }
    return {
        "schema": REAL_SCHEMA, "mode": mode, "corpus_type": "real_recorded_meetings",
        "gold_source": "independent_human_audio_review", "real_audio_gold_coverage": True,
        "text_model": text_model, "embedding_model": embed_model,
        "pass": all(thresholds.values()), "gates": thresholds,
        "counts": {"answerable_total": n_pos, "answerable_correct": correct,
                   "retrieval_recall_at_8": recall, "citation_correct": citation,
                   "negative_total": n_neg, "abstention_correct": abstain,
                   "cross_scope_leaks": leaks},
        "latency_seconds": {"p50": percentile(timings, .50), "p95": p95},
        "cases": cases,
    }


def release_decision(freeze, real_lexical=None, real_hybrid=None,
                     phase12=None, ci=None, expected_commit=None):
    """Keep Meeting knowledge status separate from full product release status."""
    def real_ok(item, mode):
        return (isinstance(item, dict) and item.get("schema") == REAL_SCHEMA and
                item.get("mode") == mode and item.get("pass") is True and
                item.get("corpus_type") == "real_recorded_meetings" and
                item.get("real_audio_gold_coverage") is True and
                isinstance(item.get("gates"), dict) and
                bool(item["gates"]) and all(v is True for v in item["gates"].values()) and
                item.get("counts", {}).get("answerable_total", 0) >= 20 and
                item.get("counts", {}).get("negative_total", 0) >= 6)
    synthetic = (isinstance(freeze, dict) and freeze.get("schema") == FREEZE_SCHEMA and
                 freeze.get("pass") is True and
                 set((freeze.get("reports") or {})) == {"lexical", "hybrid"})
    matched_models = (
        isinstance(real_lexical, dict) and isinstance(real_hybrid, dict) and
        isinstance(freeze, dict) and
        real_lexical.get("text_model") == freeze.get("text_model") and
        real_hybrid.get("text_model") == freeze.get("text_model") and
        real_lexical.get("embedding_model") is None and
        real_hybrid.get("embedding_model") == freeze.get("embedding_model") and
        isinstance(real_lexical.get("source_manifest_sha256"), str) and
        SHA256.fullmatch(real_lexical["source_manifest_sha256"]) is not None and
        real_lexical.get("source_manifest_sha256") == real_hybrid.get("source_manifest_sha256") and
        real_lexical.get("source_commit") == expected_commit and
        real_hybrid.get("source_commit") == expected_commit
    )
    real = (matched_models and real_ok(real_lexical, "lexical") and
            real_ok(real_hybrid, "hybrid"))
    phase12_ok = (isinstance(phase12, dict) and
                  phase12.get("source_commit") == expected_commit and
                  phase12.get("pass") is True and
                  bool(phase12.get("gates")) and
                  all(x is True for x in phase12["gates"].values()))
    ci_ok = (isinstance(ci, dict) and ci.get("all_required_checks_green") is True and
             isinstance(expected_commit, str) and len(expected_commit) == 40 and
             ci.get("commit_sha") == expected_commit)
    gates = {
        "synthetic_lexical_and_hybrid": synthetic,
        "real_audio_qa_lexical": real_ok(real_lexical, "lexical"),
        "real_audio_qa_hybrid": real_ok(real_hybrid, "hybrid"),
        "same_human_gold_and_models_and_commit": bool(matched_models),
        "phase12_independent_audio_and_identity": phase12_ok,
        "hosted_ci_exact_commit": ci_ok,
    }
    return {
        "schema": RELEASE_SCHEMA, "source_commit": expected_commit,
        "code_and_database_ci_pass": ci_ok,
        "local_synthetic_model_pass": synthetic,
        "knowledge_release_pass": bool(synthetic and real),
        "overall_product_release_pass": all(gates.values()),
        "gates": gates,
        "automatic_merge_or_tag": False,
        "note": "Real audio, human truth, model tests and hosted CI are separate gates.",
    }

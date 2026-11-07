"""Evidence integrity metrics. Attribution is not independent factual verification."""
from __future__ import annotations


def evidence_audit(artifact: dict, valid_segment_ids: list[int]) -> dict:
    valid = {sid for sid in valid_segment_ids if type(sid) is int}
    total = 0
    cited = 0
    invalid = 0
    duplicate_claims = 0
    seen: set[tuple[str, str]] = set()
    for field in ("actions", "decisions", "open_questions"):
        for row in artifact.get(field, []) or []:
            if not isinstance(row, dict):
                continue
            total += 1
            citations = row.get("source_ids", [])
            linked = [sid for sid in citations if type(sid) is int and sid in valid] if isinstance(citations, list) else []
            if linked:
                cited += 1
            if not isinstance(citations, list) or len(linked) != len(citations):
                invalid += 1
            key = (field, str(row.get("title", "")).casefold().strip())
            if key in seen:
                duplicate_claims += 1
            seen.add(key)
    return {"claim_count": total, "claims_with_valid_references": cited,
            "invalid_reference_claims": invalid, "duplicate_titles": duplicate_claims,
            "citation_coverage": cited / total if total else None,
            "summary_has_sentence_level_references": False,
            "independent_fact_check_performed": False}


def referenced_hits(hits: list[dict], source_numbers: object) -> list[dict]:
    if not isinstance(source_numbers, list):
        return []
    result = []
    seen: set[int] = set()
    for value in source_numbers:
        if type(value) is int and 1 <= value <= len(hits) and value not in seen:
            seen.add(value)
            result.append({k: v for k, v in hits[value - 1].items() if k != "embedding"})
    return result

"""Opt-in, local-only audit of confirmed live-caption segments.

This is a review artifact, not an ASR, diarization, or completeness score.
Unlike the aggregate replay report, it contains spoken words and potentially
identifying speaker names. Never upload audit outputs or write them by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json
import math

SCHEMA = "meeting-live-caption-segment-audit-v1"


def _labeled(item: Mapping) -> bool:
    speaker_id = item.get("speaker_id")
    return (
        type(speaker_id) is int and speaker_id > 0
        and str(item.get("speaker_name") or "").strip().lower()
        != "unidentified speaker"
    )


def _transition(before: Mapping, after: Mapping) -> str:
    previous, current = _labeled(before), _labeled(after)
    if not previous and current:
        return "unidentified_to_labeled"
    if previous and not current:
        return "labeled_to_unidentified"
    if not previous and not current:
        return "unidentified_to_other_unidentified"
    return "labeled_to_different_labeled"


def _age_seconds(now_media_ms: float | None, end_ms: object) -> float | None:
    if (type(now_media_ms) not in (int, float)
            or type(end_ms) not in (int, float)
            or not math.isfinite(now_media_ms)
            or not math.isfinite(end_ms)):
        return None
    return round(max(0, now_media_ms - end_ms) / 1000, 2)


def capture_segment_audit(
    state: dict, segments: list[dict], wall_elapsed_ms: float,
    *, media_elapsed_ms: float | None = None,
) -> None:
    """Record only changes. Replay invokes this before replacing its snapshot."""
    if not state.get("segment_audit_enabled"):
        return
    before = state["segment_audit_segments"]
    current = {}
    order = []
    for segment in segments:
        segment_id = segment.get("id")
        if type(segment_id) is not int or segment_id in current:
            continue  # The standard replay still flags invalid/duplicate IDs.
        item = {
            "segment_id": segment_id,
            "start_ms": segment.get("start_ms"),
            "end_ms": segment.get("end_ms"),
            "speaker_id": segment.get("speaker_id"),
            "speaker_name": segment.get("speaker_name"),
            "text": str(segment.get("text") or ""),
        }
        old = before.get(segment_id)
        if old is not None:
            if old["speaker_id"] != item["speaker_id"]:
                state["segment_audit_events"].append({
                    "kind": "speaker_reassignment",
                    "segment_id": segment_id,
                    "wall_elapsed_ms": round(wall_elapsed_ms, 1),
                    "media_elapsed_ms": media_elapsed_ms,
                    "segment_age_s": _age_seconds(
                        media_elapsed_ms, item["end_ms"]
                    ),
                    "transition": _transition(old, item),
                    "from_speaker_id": old["speaker_id"],
                    "to_speaker_id": item["speaker_id"],
                    "from_speaker_name": old["speaker_name"],
                    "to_speaker_name": item["speaker_name"],
                })
            elif old["speaker_name"] != item["speaker_name"]:
                state["segment_audit_events"].append({
                    "kind": "speaker_name_changed",
                    "segment_id": segment_id,
                    "wall_elapsed_ms": round(wall_elapsed_ms, 1),
                    "from_speaker_name": old["speaker_name"],
                    "to_speaker_name": item["speaker_name"],
                })
            if old["text"] != item["text"]:
                state["segment_audit_events"].append({
                    "kind": "confirmed_text_revision",
                    "segment_id": segment_id,
                    "wall_elapsed_ms": round(wall_elapsed_ms, 1),
                    "previous_text": old["text"],
                    "revised_text": item["text"],
                })
        current[segment_id] = item
        order.append(segment_id)
    for segment_id in before.keys() - current.keys():
        state["segment_audit_events"].append({
            "kind": "confirmed_segment_disappearance",
            "segment_id": segment_id,
            "wall_elapsed_ms": round(wall_elapsed_ms, 1),
            "previous_text": before[segment_id]["text"],
        })
    state["segment_audit_segments"] = current
    state["segment_audit_order"] = order


def build_segment_audit(state: Mapping, summary: Mapping) -> dict:
    """Summarize opt-in history without claiming acoustic correctness."""
    if not state.get("segment_audit_enabled"):
        raise ValueError("Segment audit was not enabled for this replay")
    final = state["segment_audit_segments"]
    order = state["segment_audit_order"]
    events = state["segment_audit_events"]
    if len(order) != len(final) or len(set(order)) != len(order):
        raise ValueError("Inconsistent final segment audit snapshot")
    if set(order) != set(state.get("segment_order", [])):
        raise ValueError("Audit does not match the final replay snapshot")
    diagnostics = summary.get("final_cluster_diagnostics") or {}
    timeline = diagnostics.get("anonymous_timeline") or {}
    evidence = {
        item.get("segment_id"): item
        for item in timeline.get("segments", [])
        if type(item.get("segment_id")) is int
    }
    missing_evidence = [
        sid for sid in order
        if evidence.get(sid, {}).get("embedding_available") is False
    ]
    flags = []
    changed_text_ids = {
        e["segment_id"] for e in events
        if e["kind"] == "confirmed_text_revision"
    }
    late_switch_ids = {
        e["segment_id"] for e in events
        if e["kind"] == "speaker_reassignment"
        and e["transition"] == "labeled_to_different_labeled"
        and e["segment_age_s"] is not None
        and e["segment_age_s"] >= 15
    }
    for position, sid in enumerate(order):
        item = final[sid]
        local = []
        if not item["text"].strip():
            local.append("empty_confirmed_text")
        if not _labeled(item):
            local.append("unidentified_speaker")
        if sid in missing_evidence:
            local.append("speaker_embedding_missing")
        if sid in changed_text_ids:
            local.append("confirmed_text_revised")
        if sid in late_switch_ids:
            local.append("late_labeled_speaker_change")
        start_ms, end_ms = item["start_ms"], item["end_ms"]
        if (type(start_ms) in (int, float)
                and type(end_ms) in (int, float)):
            if end_ms <= start_ms:
                local.append("invalid_segment_timing")
            elif end_ms - start_ms > 12000:
                local.append("long_segment_review")
            if position:
                prior_end = final[order[position - 1]]["end_ms"]
                if (type(prior_end) in (int, float)
                        and start_ms < prior_end):
                    local.append("overlap_or_timing_review")
        for flag in local:
            flags.append({"segment_id": sid, "flag": flag})
    return {
        "schema": SCHEMA,
        "source_audio": summary.get("source_audio"),
        "source_audio_duration_s": summary.get("source_audio_duration_s"),
        "final_segments": [final[sid] for sid in order],
        "revision_events": list(events),
        "review_flags": flags,
        "segments_without_speaker_embedding": missing_evidence,
        "diagnostic_timeline_truncated": timeline.get("truncated"),
        "measurement_limit": (
            "Review prompts only. A reference-backed or listening evaluation "
            "is needed to determine word omissions, merging, speaker identity, "
            "overlap correctness, and subtitle completeness."
        ),
    }


def write_private_segment_audit(path: Path, audit: Mapping) -> None:
    """Create a new JSON audit only inside the ignored Fixed195 directory."""
    base = Path("benchmark/fixed195").resolve()
    target = path.resolve()
    if not target.is_relative_to(base) or target.suffix.lower() != ".json":
        raise ValueError(
            "Private segment audit requires a .json file inside "
            "benchmark/fixed195 (gitignored)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # ASCII-safe JSON remains faithful to Unicode after parsing, and is
    # readable by Windows PowerShell 5.1 even when Get-Content is used without
    # -Encoding UTF8. Legacy bare UTF-8 reports require explicit UTF8 decoding.
    # Exclusive create ensures a later replay cannot overwrite earlier evidence.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=True, indent=2)
        stream.write("\n")

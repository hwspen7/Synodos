"""Fail closed on rapid proliferation of unverified CAM++ speaker clusters.

This guard is independent of Nemotron. The same audio might be run with the
optional diarization worker disabled, and its raw clustering labels are not
proof of distinct humans. Preserve original ASR segments and never override
manually confirmed or registered voices.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import numpy as np


def _voice(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size < 2 or not np.isfinite(result).all():
        return None
    length = float(np.linalg.norm(result))
    return result / length if length > 1e-7 else None


def repair_rapid_anonymous_births(
    segments: Iterable[dict], *,
    sample_rate: int,
    window_seconds: int = 30,
    same_voice_similarity: float = 0.80,
) -> dict[str, int]:
    """Merge only strongly matching short-lived speaker labels.

    An unverified label without equivalent voice evidence becomes unknown
    rather than an asserted fifth person. Distinct, independently repeated
    labels remain separate. This does not infer the true number of speakers.
    """
    if (sample_rate <= 0 or not 10 <= window_seconds <= 60
            or not 0.5 <= same_voice_similarity <= 1):
        raise ValueError("Invalid rapid birth guard policy")
    items = sorted(
        [s for s in segments
         if type(s.get("speaker_id")) is int and s["speaker_id"] > 0
         and type(s.get("start_sample")) is int
         and type(s.get("end_sample")) is int
         and s["end_sample"] > s["start_sample"]],
        key=lambda s: (s["start_sample"], s.get("id", 0)),
    )
    if len(items) < 3:
        return {"merged": 0, "deferred": 0}
    protected = {
        s["speaker_id"] for s in items
        if s.get("speaker_override") or s.get("matched_profile_id")
    }
    suspect: set[int] = set()
    left = 0
    for right, item in enumerate(items):
        while (items[left]["start_sample"] <
               item["start_sample"] - sample_rate * window_seconds):
            left += 1
        local = [
            s for s in items[left:right + 1]
            if s["speaker_id"] not in protected
        ]
        group_sizes: dict[int, int] = defaultdict(int)
        for s in local:
            group_sizes[s["speaker_id"]] += 1
        if (len(group_sizes) >= 3
                and sum(count == 1 for count in group_sizes.values()) >= 2):
            suspect.update(group_sizes)
    if not suspect:
        return {"merged": 0, "deferred": 0}

    groups: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        if item["speaker_id"] in suspect and item["speaker_id"] not in protected:
            groups[item["speaker_id"]].append(item)
    if len(groups) < 3:
        return {"merged": 0, "deferred": 0}

    first_label = min(groups, key=lambda sid: (
        min(s["start_sample"] for s in groups[sid]), sid
    ))
    base = groups[first_label]
    base_name = str(base[0].get("speaker_name") or f"Speaker {first_label}")
    base_voices = [v for s in base if (v := _voice(s.get("emb"))) is not None]
    merged = deferred = 0
    for label, members in groups.items():
        if label == first_label:
            continue
        independent = 0
        preceding_end = -1
        for s in sorted(members, key=lambda row: row["start_sample"]):
            if preceding_end < 0 or (
                s["start_sample"] - preceding_end >= .65 * sample_rate
            ):
                independent += 1
            preceding_end = max(preceding_end, s["end_sample"])
        # A repeatable speaker with independently separated turns is not
        # a singleton hallucination and must not be silently merged.
        if independent >= 2:
            continue
        for row in members:
            probe = _voice(row.get("emb"))
            equivalent = bool(
                probe is not None and any(
                    prior.shape == probe.shape
                    and float(np.dot(prior, probe)) >= same_voice_similarity
                    for prior in base_voices
                )
            )
            if equivalent:
                row["speaker_id"] = first_label
                row["speaker_name"] = base_name
                row["speaker_evidence_status"] = "rapid_cluster_voice_match"
                merged += 1
            else:
                row["speaker_id"] = -1
                row["speaker_name"] = "Unidentified Speaker"
                row["speaker_uncertain"] = True
                row["speaker_evidence_status"] = "rapid_cluster_birth_unconfirmed"
                deferred += 1
    return {"merged": merged, "deferred": deferred}

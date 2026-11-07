"""Priority policy when independent diarization cannot resolve a caption.

A missing Nemotron channel is *not negative evidence* against an already
source-supported CAM++ identity. Keep provenance and uncertainty explicit
rather than overwriting a supported label with Unidentified Speaker.
"""
from __future__ import annotations

import numpy as np


_PROVENANCE = frozenset({
    "direct_embedding",
    "turn_voiceprint",
    "acoustic_backfill",
    "short_probe_corroborated",
    "short_probe_known_voice",
    "multi_turn_same_voice_supported",
    "speaker_birth_voice_reused",
    "speaker_continuity_provisional",
})


def preserve_supported_camplus(record: dict) -> bool:
    """Keep an existing speaker only when its raw acoustic evidence survives.

    Manual/registered identity has a separate higher-priority policy.
    Unsupported/unknown, explicitly mixed, and uncorroborated short probes
    are not promoted here.
    """
    label, cluster = record.get("speaker_id"), record.get("cluster")
    if (type(label) is not int or label <= 0
            or type(cluster) is not int or cluster < 0
            or record.get("speaker_override")
            or record.get("matched_profile_id")):
        return False
    provenance = str(record.get("speaker_evidence_status") or "")
    if provenance not in _PROVENANCE:
        return False
    if (record.get("speaker_uncertain")
            and provenance not in {
                "speaker_continuity_provisional",
                "short_probe_known_voice",
            }):
        return False
    try:
        embedding = np.asarray(record.get("emb"), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        embedding.size >= 2
        and np.isfinite(embedding).all()
        and float(np.linalg.norm(embedding)) > 1e-7
    )

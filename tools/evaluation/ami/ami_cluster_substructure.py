"""Offline-only analysis of acoustic substructure inside merged speaker groups.

A global partition may merge two acoustically distinct speakers even
when a single-sample outlier test correctly abstains: the erroneous
group contains several consistent samples of BOTH people.

This module evaluates conservative two-way partitions of existing
groups, but it NEVER chooses the best split using human references,
changes live speaker IDs, or claims a new person. A subgroup can also
be room/microphone variation within one human speaker.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from speech.diarization.spherical_cluster import spherical_speaker_candidates

MIN_PARENT_SAMPLES = 12
MIN_CHILD_SAMPLES = 4
MIN_SUBGROUP_SILHOUETTE = 0.18


@dataclass(frozen=True)
class SubstructureCandidates:
    variants: dict[str, np.ndarray]
    details: dict


def explore_speaker_substructure(
    embeddings: np.ndarray,
    original_labels: np.ndarray, *,
    min_parent_samples: int = MIN_PARENT_SAMPLES,
    min_child_samples: int = MIN_CHILD_SAMPLES,
    min_silhouette: float = MIN_SUBGROUP_SILHOUETTE,
) -> SubstructureCandidates:
    """Propose up to one experimental two-way split per large original group.

    Each option is reported separately, as well as an all-splits option
    if more than one group qualified. Human annotations are not inputs.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    original = np.asarray(original_labels, dtype=np.int32).reshape(-1)
    if (x.ndim != 2 or not x.shape[1] or not 4 <= len(x) <= 256
            or len(x) != len(original)
            or not np.isfinite(x).all()
            or np.any(np.linalg.norm(x, axis=1) <= 1e-7)):
        raise ValueError("Invalid bounded acoustic substructure inputs")
    if (min_parent_samples < 6 or min_child_samples < 2
            or not 0.0 <= min_silhouette <= 0.9):
        raise ValueError("Invalid conservative subgroup support")
    if any(int(label) < 0 for label in original):
        raise ValueError("Expected nonnegative anonymous raw clusters")

    variants: dict[str, np.ndarray] = {"original": original.copy()}
    combined = original.copy()
    next_id = int(original.max()) + 1
    groups = []
    accepted = 0
    for label in sorted(set(map(int, original))):
        indices = np.flatnonzero(original == label)
        summary = {
            "source_cluster": label,
            "source_samples": len(indices),
            "method": "nested_spherical_k2",
            "accepted_for_review": False,
        }
        if len(indices) < max(min_parent_samples, 2 * min_child_samples):
            summary["reason"] = "insufficient_parent_samples"
            groups.append(summary)
            continue

        try:
            result = spherical_speaker_candidates(
                x[indices], max_speakers=2,
                min_silhouette=min_silhouette,
            )
        except ValueError:
            summary["reason"] = "invalid_acoustic_partition"
            groups.append(summary)
            continue
        summary["candidate_selection"] = result.details
        if result.labels is None:
            summary["reason"] = result.details.get(
                "reason", "weak_subgroup_evidence",
            )
            groups.append(summary)
            continue
        counts = np.bincount(result.labels, minlength=2)
        summary["candidate_subgroup_sizes"] = list(map(int, counts))
        if len(counts) != 2 or np.any(counts < min_child_samples):
            summary["reason"] = "insufficient_subgroup_support"
            groups.append(summary)
            continue

        # Keep the largest subgroup under the previous anonymous ID and
        # assign a NEW temporary ID only for an alternate diagnostic view.
        # A tie remains deterministic; neither subgroup is an identity.
        moving = int(np.argmin(counts))
        new_id = next_id
        next_id += 1
        child_indices = indices[result.labels == moving]
        alone = original.copy()
        alone[child_indices] = new_id
        combined[child_indices] = new_id
        key = f"split_parent_{label}"
        variants[key] = alone
        summary.update({
            "accepted_for_review": True,
            "new_temporary_cluster": new_id,
            "moved_samples": len(child_indices),
            "variant_name": key,
            "warning": (
                "A second coherent acoustic subgroup is a hypothesis, "
                "NOT proof of a different human speaker."
            ),
        })
        groups.append(summary)
        accepted += 1
    if accepted > 1:
        variants["all_review_splits"] = combined
    details = {
        "method": "nested_spherical_k2_review_only",
        "source_cluster_count": len(set(map(int, original))),
        "accepted_subdivision_count": accepted,
        "proposed_groups": groups,
        "minimum_parent_samples": min_parent_samples,
        "minimum_child_samples": min_child_samples,
        "minimum_silhouette": min_silhouette,
        "warning": (
            "Unsupervised exploratory splits of already assigned groups; "
            "a single speaker's vocal variation or environmental changes "
            "may cause false divisions. No reference labels, audio words, "
            "or speaker identity promotion enter this model."
        ),
    }
    return SubstructureCandidates(variants, details)

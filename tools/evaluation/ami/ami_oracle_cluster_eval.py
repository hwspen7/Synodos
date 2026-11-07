"""Reference-labeled evaluation of offline CAM++ solo voiceprints.

Uses human AMI reference labels ONLY for evaluation, never to choose
K, tune thresholds, seed centers, or modify real-time speaker identities.
No raw voice vectors or transcript text is written to the JSON output.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import numpy as np

from speech.diarization.adaptive_cluster import complete_link_clusters
from speech.diarization.spherical_cluster import spherical_speaker_candidates

STRICT_THRESHOLDS = (0.58, 0.62, 0.66, 0.70, 0.76)


def _choose2(n: int) -> int:
    return n * (n - 1) // 2


def score_reference_partition(
    references: list[str], assignments: np.ndarray,
) -> dict:
    """Permutation-invariant clustering scores for held-out reference samples.

    Scores every unordered pair, including one-sample speakers as negative
    pairs. Does NOT score continuous meeting audio or word-level DER.
    """
    labels = np.asarray(assignments, dtype=np.int32).reshape(-1)
    if len(labels) != len(references) or len(references) < 2:
        raise ValueError("Clustering labels do not match reference probes")
    if any(not isinstance(x, str) or not x for x in references):
        raise ValueError("Reference samples require valid anonymous IDs")
    truth_sizes = Counter(references)
    pred_sizes = Counter(map(int, labels))
    intersection = Counter(
        (reference, int(cluster))
        for reference, cluster in zip(references, labels)
    )
    tp = sum(_choose2(n) for n in intersection.values())
    predicted_pairs = sum(_choose2(n) for n in pred_sizes.values())
    same_ref_pairs = sum(_choose2(n) for n in truth_sizes.values())
    fp = predicted_pairs - tp
    fn = same_ref_pairs - tp
    precision = tp / predicted_pairs if predicted_pairs else None
    recall = tp / same_ref_pairs if same_ref_pairs else None
    pair_f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else None
    )
    total_pairs = _choose2(len(references))
    expected = predicted_pairs * same_ref_pairs / total_pairs
    maximum = (predicted_pairs + same_ref_pairs) / 2
    adjusted_rand = (
        (tp - expected) / (maximum - expected)
        if not math.isclose(maximum, expected, abs_tol=1e-12)
        else None
    )
    by_reference = []
    for speaker in sorted(truth_sizes):
        assigned = Counter(
            int(cluster)
            for ref, cluster in zip(references, labels)
            if ref == speaker
        )
        by_reference.append({
            "reference_speaker": speaker,
            "sample_count": truth_sizes[speaker],
            "cluster_contacts": [
                {"cluster": label, "samples": count}
                for label, count in sorted(
                    assigned.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "has_repeated_samples": truth_sizes[speaker] >= 2,
        })

    def maybe_round(value):
        return round(float(value), 5) if value is not None else None

    return {
        "sample_count": len(references),
        "reference_speaker_count": len(truth_sizes),
        "predicted_cluster_count": len(pred_sizes),
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "pair_precision": maybe_round(precision),
        "pair_recall": maybe_round(recall),
        "pair_f1": maybe_round(pair_f1),
        "adjusted_rand_index": maybe_round(adjusted_rand),
        "per_reference_speaker": by_reference,
        "warning": (
            "Scores only human-selected solo audio windows, not live "
            "speaker diarization DER. One-sample speakers cannot have "
            "a same-speaker positive pair and do not establish robust "
            "identity enrollment. Reference labels are evaluation-only."
        ),
    }


def provisional_isolated_members(
    embeddings: np.ndarray, assignments: np.ndarray, *,
    core_minimum_cosine: float = 0.60,
    isolation_margin: float = 0.15,
) -> tuple[np.ndarray, list[dict]]:
    """Offline-only, low-frequency outlier hypotheses inside confirmed groups.

    A candidate must be dissimilar to EVERY one of at least three other
    mutually compatible group members. This is not speaker enrollment,
    calibrated novelty detection, or a way to infer true speaker count.
    One anomalous/overlapped audio window can also trigger this rule.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(assignments, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or len(x) != len(ids) or not 4 <= len(x) <= 256:
        raise ValueError("Expected 4..256 [N,D] embeddings and [N] labels")
    if (not np.isfinite(x).all() or not 0 < core_minimum_cosine < 1
            or not 0 < isolation_margin < 1):
        raise ValueError("Invalid outlier inputs")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-7) or not np.isfinite(norms).all():
        raise ValueError("Invalid embedding norms")
    vectors = x / norms[:, None]
    hypotheses = []
    provisional = ids.copy()
    next_id = int(ids.max()) + 1
    for raw in sorted(set(map(int, ids))):
        group = np.flatnonzero(ids == raw)
        if len(group) < 4:
            continue
        best = None
        for candidate in group:
            core = group[group != candidate]
            # Require a coherent three-or-more-sample core, instead of
            # arbitrarily treating a noisy two-sample pair as a person.
            matrix = vectors[core] @ vectors[core].T
            core_cosines = matrix[np.triu_indices(len(core), k=1)]
            minimum_core = float(np.min(core_cosines))
            if minimum_core < core_minimum_cosine:
                continue
            max_cross = float(np.max(vectors[core] @ vectors[candidate]))
            margin = minimum_core - max_cross
            if margin < isolation_margin:
                continue
            if best is None or margin > best[0]:
                best = (margin, int(candidate), minimum_core, max_cross,
                        len(core))
        if best is None:
            continue
        margin, candidate, core_sim, max_cross, core_size = best
        # Hypothetical labels may increase the group count, but never
        # correspond to published IDs or retained verification templates.
        provisional[candidate] = next_id
        next_id += 1
        hypotheses.append({
            "sample_index": candidate,
            "source_cluster": raw,
            "core_samples": core_size,
            "core_minimum_cosine": round(core_sim, 5),
            "outlier_maximum_core_cosine": round(max_cross, 5),
            "cosine_gap": round(margin, 5),
            "classification": "unconfirmed_isolated_voiceprint",
        })
    return provisional, hypotheses


def evaluate_oracle_clusters(
    extracted: dict[str, list[np.ndarray]],
) -> dict:
    """Compare truly unsupervised partitions on the SAME CAM++ embeddings."""
    vectors = []
    reference = []
    for speaker in sorted(extracted):
        if not isinstance(speaker, str) or not speaker:
            raise ValueError("Invalid reference speaker ID")
        for item in extracted[speaker]:
            vec = np.asarray(item, dtype=np.float32).reshape(-1)
            if not 32 <= vec.size <= 4096 or not np.isfinite(vec).all():
                raise ValueError("Malformed CAM++ voiceprint")
            norm = np.linalg.norm(vec)
            if not np.isfinite(norm) or norm <= 1e-7:
                raise ValueError("Invalid CAM++ voiceprint norm")
            vectors.append(vec / norm)
            reference.append(speaker)
    if len(vectors) < 4:
        return {
            "available": False,
            "reason": "fewer_than_four_valid_solo_voiceprints",
            "successful_voiceprints": len(vectors),
        }
    dimensions = {v.shape for v in vectors}
    if len(dimensions) != 1:
        raise ValueError("Inconsistent CAM++ embedding shapes")
    matrix = np.stack(vectors)
    if len(matrix) > 256:
        return {
            "available": False,
            "reason": "more_than_256_voiceprints",
            "successful_voiceprints": len(matrix),
        }
    methods = []
    for threshold in STRICT_THRESHOLDS:
        try:
            labels = complete_link_clusters(
                matrix, threshold=threshold,
            )
            methods.append({
                "method": "complete_link",
                "cosine_threshold": threshold,
                "available": True,
                "score": score_reference_partition(reference, labels),
            })
        except ValueError:
            methods.append({
                "method": "complete_link",
                "cosine_threshold": threshold,
                "available": False,
                "reason": "rejected_or_excessive_fragmentation",
            })
    candidate = spherical_speaker_candidates(matrix)
    if candidate.labels is not None:
        methods.append({
            "method": "spherical_auto",
            "available": True,
            "selection": candidate.details,
            "score": score_reference_partition(reference, candidate.labels),
        })
        # Additional non-production hypothesis: isolated singleton samples
        # may be separate people, or may simply be acoustic artifacts.
        provisional, flagged = provisional_isolated_members(
            matrix, candidate.labels,
        )
        methods.append({
            "method": "spherical_provisional_outliers",
            "available": True,
            "provisional_isolated_count": len(flagged),
            "unconfirmed_samples": flagged,
            "score": score_reference_partition(reference, provisional),
            "warning": (
                "Experimental reference-only split of an unusually "
                "isolated single embedding. Do not assign this new "
                "cluster a published speaker ID. A corrupted or mixed "
                "voiceprint is indistinguishable from a genuinely new "
                "person without additional independent audio evidence."
            ),
        })
    else:
        methods.append({
            "method": "spherical_auto",
            "available": False,
            "selection": candidate.details,
            "reason": candidate.details.get("reason", "abstained"),
        })
    return {
        "available": True,
        "successful_voiceprints": len(matrix),
        "reference_speaker_count_for_evaluation_only": len(set(reference)),
        "methods": methods,
        "warning": (
            "No method is informed of the reference speaker count when "
            "forming acoustic groups. Ground-truth IDs are used afterward "
            "solely to evaluate anonymous sample partitions. These are "
            "oracle-clean audio windows, not production speaker embeddings "
            "or live DER; do not tune on this meeting and report the same "
            "recording as independent validation."
        ),
    }

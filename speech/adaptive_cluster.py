"""Bounded conservative speaker clustering without FunASR's <20-vector collapse.

This is an independent acoustic *candidate* clustering implementation.
No transcript, person names or pre-enrolled identity is used. A complete-
link cosine constraint avoids greedy centroid chaining across speakers.

It cannot resolve simultaneous people from one mixed embedding or infer
the number of real people from absent voiceprints.
"""
from __future__ import annotations

import math
import numpy as np

MAX_EMBEDDINGS = 256
MAX_CLUSTERS = 32


def complete_link_clusters(
    embeddings: np.ndarray, *, threshold: float = 0.76,
    max_clusters: int = MAX_CLUSTERS,
) -> np.ndarray:
    """Deterministic complete-link cosine clustering for up to 256 vectors.

    Two groups merge only if *every* cross-pair has cosine similarity at
    least threshold. This is a conservative acoustic partition; do not use
    cluster numbers as persistent speaker identities.
    """
    if not (math.isfinite(threshold) and 0.1 < threshold < 0.99):
        raise ValueError("Invalid cosine threshold")
    if not 1 <= max_clusters <= MAX_CLUSTERS:
        raise ValueError("Invalid speaker cluster budget")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] > MAX_EMBEDDINGS or not array.shape[1]:
        raise ValueError("Expected bounded [N,D] embeddings")
    n = array.shape[0]
    if not n:
        return np.empty(0, dtype=np.int32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite speaker embedding")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-7):
        raise ValueError("Zero speaker embedding")
    vectors = array / norms[:, None]
    sims = np.clip(vectors @ vectors.T, -1.0, 1.0)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None
        best_sim = threshold
        for i, left in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                # Complete linkage: one disagreeing/overlapping voice
                # window cannot bridge acoustically distinct speakers.
                score = float(np.min(sims[np.ix_(left, clusters[j])]))
                if score >= best_sim:
                    if score > best_sim or best is None:
                        best = (i, j)
                        best_sim = score
        if best is None:
            break
        i, j = best
        clusters[i].extend(clusters.pop(j))
        clusters[i].sort()
    if len(clusters) > max_clusters:
        # Refuse to invent an over-capacity assignment; higher-level
        # caller can retain the established clustering unchanged.
        raise ValueError("Insufficient acoustic agreement for cluster budget")
    clusters.sort(key=lambda group: min(group))
    labels = np.empty(n, dtype=np.int32)
    for cluster_id, members in enumerate(clusters):
        labels[members] = cluster_id
    return labels


def choose_cluster_labels(
    embeddings: np.ndarray, funasr_labels: np.ndarray | None = None, *,
    mode: str = "bootstrap", bootstrap_threshold: float = 0.76,
) -> tuple[np.ndarray, str]:
    """Safely bridge low-N FunASR behavior; keep standard higher-N default.

    bootstrap: complete linkage only under 20 examples, normal FunASR above.
    candidate: complete linkage up to 256 vectors. For >256 return the
    normal clustering and report non-evaluability (no partial-label mixes).
    legacy: always use current FunASR output.

    A candidate split is a hypothesis, not proof of a distinct person.
    """
    embs = np.asarray(embeddings, dtype=np.float32)
    if embs.ndim != 2:
        raise ValueError("Expected two-dimensional voiceprints")
    n = len(embs)
    if mode not in ("legacy", "bootstrap", "candidate"):
        raise ValueError("Unrecognized speaker clustering strategy")
    if mode == "legacy" or (mode == "bootstrap" and n >= 20) or n > MAX_EMBEDDINGS:
        if funasr_labels is None:
            raise ValueError("FunASR labels required for this strategy")
        labels = np.asarray(funasr_labels, dtype=np.int32).reshape(-1)
        if len(labels) != n:
            raise ValueError("FunASR label count mismatch")
        return labels, ("legacy" if mode != "candidate" else "candidate_over_capacity")
    labels = complete_link_clusters(embs, threshold=bootstrap_threshold)
    return labels, ("bootstrap" if n < 20 else "candidate")

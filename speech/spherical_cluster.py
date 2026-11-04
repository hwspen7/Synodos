"""Experimental bounded spherical-k-means speaker candidates with model selection.

Unlike complete-link thresholding, every sample does not have to match
every other sample of a speaker. The model-selection score is heuristic:
it is NOT a calibrated likelihood, a known number of people, or DER.

No reference labels, meeting text, stored identities, or user names are
used for selection. All inputs/outputs remain in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

MAX_SAMPLES = 256
MAX_SPEAKERS = 8


@dataclass(frozen=True)
class SphericalCandidate:
    labels: np.ndarray | None
    details: dict


def _vectors(embeddings: np.ndarray) -> np.ndarray:
    data = np.asarray(embeddings, dtype=np.float32)
    if (data.ndim != 2 or not 2 <= len(data) <= MAX_SAMPLES
            or not data.shape[1] or not np.isfinite(data).all()):
        raise ValueError("Expected 2..256 finite [N,D] speaker vectors")
    norms = np.linalg.norm(data, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-7):
        raise ValueError("Invalid speaker vector norm")
    return data / norms[:, None]


def _run_kmeans(vectors: np.ndarray, k: int, first: int) -> np.ndarray | None:
    """Farthest-first deterministic initialization, then spherical updates."""
    n = len(vectors)
    seeds = [first % n]
    closest = vectors @ vectors[seeds[0]]
    for _ in range(1, k):
        next_seed = int(np.argmin(closest))
        if next_seed in seeds:
            return None
        seeds.append(next_seed)
        closest = np.maximum(closest, vectors @ vectors[next_seed])
    centers = vectors[seeds].copy()
    previous = None
    for _ in range(30):
        labels = np.argmax(vectors @ centers.T, axis=1).astype(np.int32)
        if np.array_equal(labels, previous):
            break
        previous = labels.copy()
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            return None
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, vectors)
        norms = np.linalg.norm(sums, axis=1)
        if np.any(norms <= 1e-8):
            return None
        centers = sums / norms[:, None]
    return labels


def _silhouette(distance: np.ndarray, labels: np.ndarray, k: int) -> float:
    """True sample-level silhouette under cosine distance."""
    counts = np.bincount(labels, minlength=k)
    if np.any(counts < 2):
        return -1.0
    masks = (labels[:, None] == np.arange(k)[None, :]).astype(np.float64)
    mean_distances = (distance @ masks) / counts[None, :]
    idx = np.arange(len(labels))
    own = (distance @ masks)[idx, labels] / (counts[labels] - 1)
    mean_distances[idx, labels] = np.inf
    nearest = np.min(mean_distances, axis=1)
    denom = np.maximum(own, nearest)
    scores = np.divide(nearest - own, denom,
                       out=np.zeros_like(own), where=denom > 1e-10)
    return float(np.mean(scores))


def spherical_speaker_candidates(
    embeddings: np.ndarray, *, max_speakers: int = MAX_SPEAKERS,
    min_silhouette: float = 0.15,
) -> SphericalCandidate:
    """Return one supported candidate partition or abstain.

    Avoids forced k=4, fixed two-person limits, and false singleton
    identities. At least two samples per proposed speaker are required.
    Heuristic scores must be evaluated on separate real annotated audio.
    """
    if not 2 <= max_speakers <= MAX_SPEAKERS:
        raise ValueError("max_speakers must be 2..8")
    if not 0 <= min_silhouette <= 0.9:
        raise ValueError("min_silhouette must be in [0,0.9]")
    vectors = _vectors(embeddings)
    n = len(vectors)
    top_k = min(max_speakers, n // 2)
    if top_k < 2:
        return SphericalCandidate(None, {
            "available": False, "reason": "insufficient_repeated_samples",
            "eligible_embeddings": n, "method": "spherical_silhouette",
        })
    sim = np.clip(vectors @ vectors.T, -1.0, 1.0)
    dist = 1.0 - sim.astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    trials = []
    best = None
    starts = sorted({0, n // 3, (2 * n) // 3, n - 1})
    for k in range(2, top_k + 1):
        best_k = None
        for start in starts:
            labels = _run_kmeans(vectors, k, start)
            if labels is None:
                continue
            counts = np.bincount(labels, minlength=k)
            if np.any(counts < 2):
                continue
            silhouette = _silhouette(dist, labels, k)
            # Small complexity penalty discourages gratuitous speaker splits.
            score = silhouette - 0.018 * (k - 2)
            if best_k is None or score > best_k[0]:
                best_k = (score, silhouette, labels, counts)
        if best_k is None:
            trials.append({
                "k": k, "usable": False, "reason": "singleton_or_empty_group",
            })
            continue
        score, silhouette, labels, counts = best_k
        trials.append({
            "k": k, "usable": True,
            "silhouette": round(silhouette, 5),
            "selection_score": round(score, 5),
            "smallest_cluster": int(min(counts)),
        })
        if best is None or score > best[0]:
            best = (score, silhouette, k, labels)
    summary = {
        "method": "spherical_silhouette",
        "eligible_embeddings": n,
        "max_speakers_evaluated": top_k,
        "candidates": trials,
        "min_silhouette": min_silhouette,
        "warning": (
            "Unsupervised acoustic grouping. A high silhouette does not "
            "prove speaker identity, a true speaker count or lower DER."
        ),
    }
    if best is None:
        return SphericalCandidate(None, {
            **summary, "available": False,
            "reason": "all_candidate_clusters_undersupported",
        })
    score, silhouette, k, labels = best
    if silhouette < min_silhouette:
        return SphericalCandidate(None, {
            **summary, "available": False,
            "reason": "weak_acoustic_separation",
            "best_candidate_k": k,
        })
    # Stable IDs are assigned by processor after this raw partition.
    order = sorted(set(labels), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    mapping = {int(label): index for index, label in enumerate(order)}
    final = np.array([mapping[int(label)] for label in labels], dtype=np.int32)
    return SphericalCandidate(final, {
        **summary, "available": True,
        "chosen_cluster_count": k,
        "selection_score": round(score, 5),
        "silhouette": round(silhouette, 5),
        "smallest_cluster": int(np.bincount(final).min()),
    })

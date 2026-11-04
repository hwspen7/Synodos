"""Cross-language speaker ID stability tools, separate from ASR language detection."""
from __future__ import annotations
import numpy as np


class SpeakerCentroidTracker:
    """Incremental cosine similarity matching, independent of spoken language."""
    def __init__(self, threshold: float = 0.72, max_speakers: int = 64):
        if not 0 < threshold < 1 or max_speakers <= 0:
            raise ValueError("Invalid speaker tracker limits")
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}

    def assign(self, embedding: np.ndarray | None) -> int | None:
        if embedding is None:
            return None
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        vec = vec / norm
        similarities = [(float(np.dot(vec, c)), sid) for sid, c in self.centroids.items() if c.shape == vec.shape]
        if similarities and max(similarities)[0] >= self.threshold:
            sid = max(similarities)[1]
            n = self.counts[sid]
            combined = self.centroids[sid] * n + vec
            self.centroids[sid] = combined / max(np.linalg.norm(combined), 1e-6)
            self.counts[sid] += 1
            return sid
        if len(self.centroids) >= self.max_speakers:
            return None
        sid = max(self.centroids, default=0) + 1
        self.centroids[sid] = vec
        self.counts[sid] = 1
        return sid

    def seed(self, speaker_id: int, embedding: np.ndarray) -> None:
        """Rebuild a speaker prototype from every prior labeled embedding.

        The tracker is reconstructed for each incremental clustering pass.
        Ignoring subsequent examples for the same label leaves an early,
        possibly atypical, utterance as the permanent centroid and causes
        unnecessary identity proliferation during long meetings.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if (speaker_id < 1 or not np.isfinite(norm) or norm < 1e-6
                or not np.isfinite(vec).all()):
            return
        unit = vec / norm
        current = self.centroids.get(speaker_id)
        if current is None:
            self.centroids[speaker_id] = unit
            self.counts[speaker_id] = 1
            return
        if current.shape != unit.shape:
            return
        n = self.counts[speaker_id]
        combined = current * n + unit
        combined_norm = float(np.linalg.norm(combined))
        if np.isfinite(combined_norm) and combined_norm >= 1e-6:
            self.centroids[speaker_id] = combined / combined_norm
            self.counts[speaker_id] = n + 1


def assign_acoustic_anchor_clusters(
    records: list[dict],
    candidate_indices: list[int],
    cluster_model,
    sample_rate: int,
    *,
    min_anchor_s: float = 1.2,
    short_match_similarity: float = 0.72,
    short_match_margin: float = 0.08,
) -> dict[str, int]:
    """Experimental two-stage diarization without an assumed speaker count.

    Cluster reliable, sufficiently long voice segments first. Short and
    uncertain segments may borrow an existing raw cluster only when their
    embedding sufficiently and unambiguously agrees with its centroid.
    Short segments never create an identity. No text or voiceprints leave
    this in-memory processor.

    This opt-in experiment must be measured against human speaker turns
    before it is suitable as the production default.
    """
    from sessions.speaker_fallback import UNIDENTIFIED_CLUSTER

    if sample_rate < 1000 or not 0.4 <= min_anchor_s <= 10:
        raise ValueError("Invalid acoustic anchor duration")
    if not 0 < short_match_similarity < 1:
        raise ValueError("Invalid short segment similarity threshold")
    if not 0 <= short_match_margin < 1:
        raise ValueError("Invalid short segment ambiguity margin")

    vectors: dict[int, np.ndarray] = {}
    dimension = None
    for index in candidate_indices:
        record = records[index]
        record["cluster"] = UNIDENTIFIED_CLUSTER
        try:
            vec = np.asarray(record.get("emb"), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            continue
        if not vec.size or not np.isfinite(vec).all():
            continue
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-6:
            continue
        if dimension is None:
            dimension = vec.size
        if vec.size != dimension:
            continue
        vectors[index] = vec / norm

    minimum_samples = round(min_anchor_s * sample_rate)
    anchors = [
        index for index in candidate_indices
        if index in vectors
        and (records[index]["end_sample"] - records[index]["start_sample"])
        >= minimum_samples
        and not records[index].get("speaker_uncertain", False)
        and not records[index].get("speaker_override", False)
    ]
    if not anchors:
        return {"anchor_segments": 0, "short_assigned": 0,
                "unassigned": len(candidate_indices), "raw_clusters": 0}

    # Some spectral clustering backends require at least two observations.
    # One trusted voice prototype is enough for provisional one-cluster
    # matching without invoking their eigensolver on a 1xD matrix.
    raw_labels = (
        np.array([0], dtype=int) if len(anchors) == 1
        else np.asarray(
            cluster_model(np.stack([vectors[i] for i in anchors])),
        ).reshape(-1)
    )
    if len(raw_labels) != len(anchors):
        raise ValueError("Speaker cluster model returned wrong label count")
    cluster_sums: dict[int, np.ndarray] = {}
    for index, raw in zip(anchors, raw_labels):
        label = int(raw)
        if label < 0:
            raise ValueError("Speaker cluster model returned negative raw ID")
        records[index]["cluster"] = label
        cluster_sums[label] = (
            cluster_sums.get(label, np.zeros(dimension, dtype=np.float64))
            + vectors[index].astype(np.float64)
        )
    centers = {}
    for label, value in cluster_sums.items():
        norm = float(np.linalg.norm(value))
        if norm > 1e-8:
            centers[label] = value / norm

    assigned = 0
    for index in candidate_indices:
        # The upstream acoustic turn-splitter explicitly marks possible
        # mixed-speaker intervals uncertain; a high mean cosine match
        # must not override that evidence of ambiguity.
        if (index not in vectors or index in anchors
                or records[index].get("speaker_uncertain", False)):
            continue
        similarities = sorted(
            ((float(np.dot(vectors[index], center)), label)
             for label, center in centers.items()),
            reverse=True,
        )
        if not similarities:
            continue
        best_sim, best_label = similarities[0]
        runner_up = similarities[1][0] if len(similarities) > 1 else -1.0
        if (best_sim >= short_match_similarity
                and best_sim - runner_up >= short_match_margin):
            records[index]["cluster"] = best_label
            assigned += 1

    return {
        "anchor_segments": len(anchors),
        "short_assigned": assigned,
        "unassigned": sum(
            records[i]["cluster"] == UNIDENTIFIED_CLUSTER
            for i in candidate_indices
        ),
        "raw_clusters": len(cluster_sums),
    }

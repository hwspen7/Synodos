"""Corroborate new anonymous acoustic speakers against voice embeddings.

Nemotron channel indices are not speaker identities. A channel may flicker
during continuous speech. Never create a new visible person from a new
channel alone, and never treat transcript text as voice evidence.
"""
from __future__ import annotations

import math
import numpy as np

from sessions.stable_speaker_channels import StableSpeakerChannels


def _unit(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError):
        return None
    if vector.size < 2 or not np.isfinite(vector).all():
        return None
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-7:
        return None
    return vector / length


class CorroboratedSpeakerChannels:
    """Session-local gate for assigning meeting speaker IDs.

    An already-established raw channel retains its ID. A new raw channel can
    reuse an existing ID if its embedding is close to established voice
    evidence. A truly new speaker requires repeated, consistent, strongly
    *different* voice embeddings as well as model activity.

    Uncertain candidates return None and must be published as unidentified,
    not as a CAM++ provisional speaker number.
    """

    def __init__(
        self, offset: int = 0, *,
        same_voice_similarity: float = 0.78,
        distinct_voice_max_similarity: float = 0.48,
        distinct_candidate_similarity: float = 0.70,
        new_voice_min_ms: int = 3000,
        new_voice_min_turns: int = 2,
    ) -> None:
        if (not 0 <= distinct_voice_max_similarity < same_voice_similarity <= 1
                or not 0 < distinct_candidate_similarity <= 1
                or not 500 <= new_voice_min_ms <= 30000
                or not 2 <= new_voice_min_turns <= 6):
            raise ValueError("Invalid corroborated speaker thresholds")
        self._channels = StableSpeakerChannels(offset=offset)
        self.same_voice_similarity = same_voice_similarity
        self.distinct_voice_max_similarity = distinct_voice_max_similarity
        self.distinct_candidate_similarity = distinct_candidate_similarity
        self.new_voice_min_ms = new_voice_min_ms
        self.new_voice_min_turns = new_voice_min_turns
        self._references: dict[int, list[np.ndarray]] = {}
        self._candidates: dict[int, dict[int, tuple[int, np.ndarray]]] = {}

    @property
    def offset(self) -> int:
        return self._channels.offset

    @property
    def labels(self) -> dict[int, int]:
        return self._channels.labels

    def _remember(self, display: int, vector: np.ndarray | None) -> None:
        if vector is None:
            return
        entries = self._references.setdefault(display, [])
        if not entries or max(
            (float(np.dot(item, vector)) for item in entries
             if item.shape == vector.shape), default=-1.0
        ) < 0.995:
            entries.append(vector)
            if len(entries) > 8:
                del entries[0]

    def _scores(self, vector: np.ndarray) -> list[tuple[int, float]]:
        ranked = []
        for display, refs in self._references.items():
            sims = [
                float(np.dot(vector, item))
                for item in refs if item.shape == vector.shape
            ]
            if sims:
                ranked.append((display, max(sims)))
        return sorted(ranked, key=lambda row: (-row[1], row[0]))

    def observe(
        self, channel: int, segment_id: int, evidence_ms: int,
        embedding: object = None, *,
        reserved_labels: set[int] | None = None,
    ) -> int | None:
        if (type(channel) is not int or not 1 <= channel <= 8
                or type(segment_id) is not int or segment_id <= 0
                or type(evidence_ms) is not int or evidence_ms <= 0):
            return None
        vector = _unit(embedding)
        existing = self._channels.existing(channel)
        if existing is not None:
            scores = self._scores(vector) if vector is not None else []
            # A strongly incompatible voice on an already-mapped channel
            # is unknown rather than forcibly assigned to that person.
            if (vector is not None and scores
                    and scores[0][0] != existing
                    and scores[0][1] >= self.same_voice_similarity):
                return None
            self._remember(existing, vector)
            return existing

        if not self._channels.labels:
            label = self._channels.resolve(
                channel, evidence_ms, reserved_labels=reserved_labels,
            )
            if label is not None:
                self._remember(label, vector)
            return label

        if vector is None:
            return None
        scores = self._scores(vector)
        if not scores:
            # Without a known person's embedding there is no acoustic basis
            # for deciding whether another channel is a new human.
            return None
        best_label, similarity = scores[0]
        if (similarity >= self.same_voice_similarity
                and (len(scores) == 1 or similarity - scores[1][1] >= 0.08)):
            self._channels.bind(channel, best_label)
            self._remember(best_label, vector)
            self._candidates.pop(channel, None)
            return best_label
        if similarity > self.distinct_voice_max_similarity:
            return None

        turns = self._candidates.setdefault(channel, {})
        # The same transcript segment is evaluated repeatedly while audio
        # arrives. Replace its evidence instead of counting it as new turns.
        turns[segment_id] = (evidence_ms, vector)
        if len(turns) < self.new_voice_min_turns:
            return None
        if sum(duration for duration, _ in turns.values()) < self.new_voice_min_ms:
            return None
        vectors = [item for _, item in turns.values()]
        if not all(
            float(np.dot(vectors[0], other)) >= self.distinct_candidate_similarity
            for other in vectors[1:]
        ):
            return None
        # Recheck against ALL known references just before allocating.
        if any(self._scores(candidate)[0][1] > self.distinct_voice_max_similarity
               for candidate in vectors):
            return None
        label = self._channels.resolve(
            channel, sum(duration for duration, _ in turns.values()),
            reserved_labels=reserved_labels,
        )
        if label is not None:
            for candidate in vectors:
                self._remember(label, candidate)
            self._candidates.pop(channel, None)
        return label

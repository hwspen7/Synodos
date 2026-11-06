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
        # Medium-distance voices require much more corroboration than voices
        # demonstrably unlike every existing identity. This avoids the old
        # 0.48--0.78 dead zone without equating one new Nemotron channel to a
        # new person.
        self.medium_distinct_max_similarity = 0.70
        self.medium_candidate_similarity = 0.84
        self.medium_min_turns = 5
        self.medium_min_ms = 9000
        self._last_gate_reason: dict[int, str] = {}
        self._references: dict[int, list[np.ndarray]] = {}
        self._candidates: dict[int, dict[int, tuple[int, np.ndarray]]] = {}

    @property
    def offset(self) -> int:
        return self._channels.offset

    @property
    def labels(self) -> dict[int, int]:
        return self._channels.labels

    def gate_reason(self, channel: int) -> str:
        """Anonymous, vector-free explanation for the last channel decision."""
        return self._last_gate_reason.get(channel, "unavailable")

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

    def match_known(self, embedding: object) -> int | None:
        """Reuse an established speaker only with strong voice evidence.

        Unlike channel allocation, this does not create or bind a new channel.
        Useful when a later Nemotron activity frame is missing or ambiguous
        but an independent CAM++ embedding is available.
        """
        vector = _unit(embedding)
        if vector is None:
            return None
        ranked = self._scores(vector)
        if not ranked:
            return None
        best_label, best_score = ranked[0]
        if best_score < self.same_voice_similarity:
            return None
        if len(ranked) > 1 and best_score - ranked[1][1] < 0.08:
            return None
        return best_label

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
            self._last_gate_reason[channel] = "existing_channel"
            own = [
                float(np.dot(vector, reference))
                for reference in self._references.get(existing, [])
                if vector is not None and vector.shape == reference.shape
            ]
            if own and max(own) < self.distinct_voice_max_similarity:
                self._last_gate_reason[channel] = "existing_channel_voice_conflict"
                # The model reused an existing channel for acoustically
                # incompatible speech. Never force the old person's label.
                return None
            scores = self._scores(vector) if vector is not None else []
            # A strongly incompatible voice on an already-mapped channel
            # is unknown rather than forcibly assigned to that person.
            if (vector is not None and scores
                    and scores[0][0] != existing
                    and scores[0][1] >= self.same_voice_similarity):
                self._last_gate_reason[channel] = "existing_channel_matches_other_voice"
                return None
            self._remember(existing, vector)
            return existing

        if not self._channels.labels:
            label = self._channels.resolve(
                channel, evidence_ms, reserved_labels=reserved_labels,
            )
            if label is not None:
                self._remember(label, vector)
                self._last_gate_reason[channel] = "first_channel_confirmed"
            else:
                self._last_gate_reason[channel] = "first_channel_insufficient_activity"
            return label

        if vector is None:
            self._last_gate_reason[channel] = "missing_voice_embedding"
            return None
        scores = self._scores(vector)
        if not scores:
            # Without a known person's embedding there is no acoustic basis
            # for deciding whether another channel is a new human.
            self._last_gate_reason[channel] = "missing_known_voice_references"
            return None
        best_label, similarity = scores[0]
        if (similarity >= self.same_voice_similarity
                and (len(scores) == 1 or similarity - scores[1][1] >= 0.08)):
            self._channels.bind(channel, best_label)
            self._remember(best_label, vector)
            self._candidates.pop(channel, None)
            self._last_gate_reason[channel] = "existing_voice_reused"
            return best_label

        # A similarity in (0.48, 0.78) used to drop candidate evidence
        # outright. Keep a conservative middle band, but do not allocate
        # another person from an equivocal near-match to an existing voice.
        if similarity > self.medium_distinct_max_similarity:
            # A near-match to a known voice invalidates this proposed birth,
            # rather than leaving stale strong-distinct candidates behind.
            self._candidates.pop(channel, None)
            self._last_gate_reason[channel] = "candidate_similar_to_existing_voice"
            return None

        turns = self._candidates.setdefault(channel, {})
        # One segment ID can be evaluated repeatedly during streaming;
        # it must never count as additional independent evidence.
        turns[segment_id] = (evidence_ms, vector)
        while len(turns) > 32:
            turns.pop(next(iter(turns)))
        # A stray clip must not poison a channel forever. Select a
        # mutually consistent, duration-weighted majority, and never use
        # an equally supported rival voice to confirm a new identity.
        members = list(turns.items())
        scores = [self._scores(candidate) for _, (_, candidate) in members]
        similarities = np.array([
            [float(np.dot(left, right)) if left.shape == right.shape else -1.0
             for _, (_, right) in members]
            for _, (_, left) in members
        ])
        order = sorted(range(len(members)), key=lambda i: (-members[i][1][0], members[i][0]))
        total_ms = sum(duration for _, (duration, _) in members)
        winner = None
        for medium, limit, minimum, count, duration in (
            (False, self.distinct_voice_max_similarity,
             self.distinct_candidate_similarity, self.new_voice_min_turns, self.new_voice_min_ms),
            (True, self.medium_distinct_max_similarity,
             self.medium_candidate_similarity, self.medium_min_turns, self.medium_min_ms),
        ):
            eligible = [i for i in order if scores[i] and scores[i][0][1] <= limit]
            for anchor in eligible:
                cohort = [anchor]
                for i in eligible:
                    if i != anchor and all(similarities[i, j] >= minimum for j in cohort):
                        cohort.append(i)
                support = sum(members[i][1][0] for i in cohort)
                if (len(cohort) < count or support < duration
                        or len(cohort) * 2 <= len(members) or support * 2 <= total_ms):
                    continue
                if winner is None or support > winner[0]:
                    winner = (support, cohort, medium)
        if winner is None:
            medium = any(score and score[0][1] > self.distinct_voice_max_similarity
                         for score in scores)
            required_turns = self.medium_min_turns if medium else self.new_voice_min_turns
            required_ms = self.medium_min_ms if medium else self.new_voice_min_ms
            self._last_gate_reason[channel] = (
                "candidate_needs_more_turns" if len(turns) < required_turns
                else "candidate_needs_more_audio" if total_ms < required_ms
                else "candidate_voice_inconsistent"
            )
            return None
        duration_ms, cohort, medium_distance = winner
        vectors = [members[i][1][1] for i in cohort]
        current_supported = any(members[i][0] == segment_id for i in cohort)
        label = self._channels.resolve(
            channel, duration_ms, reserved_labels=reserved_labels,
        )
        if label is not None:
            for candidate in vectors:
                self._remember(label, candidate)
            self._candidates.pop(channel, None)
            self._last_gate_reason[channel] = (
                "medium_voice_confirmed" if medium_distance
                else "distinct_voice_confirmed"
            )
        else:
            self._last_gate_reason[channel] = "channel_minimum_activity_pending"
        if label is not None and not current_supported:
            self._last_gate_reason[channel] = "candidate_voice_outlier"
            return None
        return label

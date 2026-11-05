"""Stable display IDs for Nemotron's anonymous acoustic channels.

Nemotron channels are model-local indices, not speaker counts or person IDs.
The fifth model channel can be the first actual voice in a fresh recording.
We allocate the first sufficiently supported channel the first available
meeting label, without changing an ID after it is published.
"""
from __future__ import annotations


class StableSpeakerChannels:
    def __init__(self, offset: int = 0, minimum_birth_ms: int = 900):
        if type(offset) is not int or offset < 0:
            raise ValueError("Speaker label offset must be nonnegative")
        if type(minimum_birth_ms) is not int or not 200 <= minimum_birth_ms <= 5000:
            raise ValueError("Invalid first-speaker evidence threshold")
        self.offset = offset
        self.minimum_birth_ms = minimum_birth_ms
        self._labels: dict[int, int] = {}

    @property
    def labels(self) -> dict[int, int]:
        return dict(self._labels)

    def existing(self, channel: int) -> int | None:
        if type(channel) is not int or not 1 <= channel <= 8:
            return None
        return self._labels.get(channel)

    def resolve(
        self, channel: int, evidence_ms: int,
        reserved_labels: set[int] | None = None,
    ) -> int | None:
        """Never create identities from brief activity spikes.

        Allocation follows the first accepted caption, not model channel
        index or a guessed human identity. Existing labels can be reused
        for short, independently supported acknowledgements.
        """
        if type(channel) is not int or not 1 <= channel <= 8:
            return None
        prior = self._labels.get(channel)
        if prior is not None:
            return prior
        if type(evidence_ms) is not int or evidence_ms < self.minimum_birth_ms:
            return None
        reserved = {
            value for value in (reserved_labels or set())
            if type(value) is int and value > 0
        }
        used = set(self._labels.values()) | reserved
        label = self.offset + 1
        while label in used:
            label += 1
        self._labels[channel] = label
        return label

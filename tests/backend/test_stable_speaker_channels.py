"""Unit and session integration tests for stable Nemotron speaker labels."""
from __future__ import annotations

import unittest

from sessions.channels.stable_speaker_channels import StableSpeakerChannels


class StableSpeakerChannelsTests(unittest.TestCase):
    def test_fifth_raw_channel_is_first_visible_voice_in_new_meeting(self):
        labels = StableSpeakerChannels(offset=0)
        self.assertEqual(labels.resolve(5, 1500), 1)
        self.assertEqual(labels.resolve(5, 250), 1)
        self.assertEqual(labels.resolve(2, 1200), 2)
        self.assertEqual(labels.labels, {5: 1, 2: 2})

    def test_short_phantom_does_not_consume_speaker_number(self):
        labels = StableSpeakerChannels()
        self.assertIsNone(labels.resolve(6, 150))
        self.assertIsNone(labels.resolve(7, 899))
        self.assertEqual(labels.resolve(5, 900), 1)
        self.assertEqual(labels.labels, {5: 1})

    def test_resume_allocates_after_history_without_reusing_old_id(self):
        labels = StableSpeakerChannels(offset=4)
        self.assertEqual(labels.resolve(5, 2000), 5)
        self.assertEqual(labels.resolve(1, 2000), 6)

    def test_identity_collision_is_avoided(self):
        labels = StableSpeakerChannels()
        self.assertEqual(labels.resolve(5, 2000, {1, 3}), 2)
        self.assertEqual(labels.resolve(5, 200, {1, 3}), 2)
        self.assertEqual(labels.resolve(2, 1100, {1, 3}), 4)

    def test_invalid_channels_cannot_allocate_labels(self):
        labels = StableSpeakerChannels()
        for channel in (-1, 0, 9, None, True, 1.5):
            self.assertIsNone(labels.resolve(channel, 2000))
        self.assertEqual(labels.labels, {})

    def test_public_map_is_a_copy(self):
        labels = StableSpeakerChannels()
        labels.resolve(5, 1400)
        external = labels.labels
        external[5] = 99
        self.assertEqual(labels.existing(5), 1)


if __name__ == "__main__":
    unittest.main()

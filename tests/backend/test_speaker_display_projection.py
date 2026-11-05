"""Integration of stable speaker labels with acoustic/word projections."""
from __future__ import annotations

import unittest

from sessions.stable_speaker_channels import StableSpeakerChannels
from speech.content_first_projection import project_confirmed_caption
from speech.overlap_content_risk import overlap_content_risk


class SpeakerDisplayProjectionTests(unittest.TestCase):
    def test_raw_channel_five_projects_as_first_meeting_speaker(self):
        labels = StableSpeakerChannels()
        self.assertEqual(labels.resolve(5, 1200), 1)
        self.assertEqual(labels.resolve(2, 1200), 2)
        projection = project_confirmed_caption(
            text="Hello world",
            words=[
                {"text": "Hello ", "start_ms": 0, "end_ms": 1200},
                {"text": "world", "start_ms": 1200, "end_ms": 2400},
            ],
            caption_start_ms=0,
            caption_end_ms=2400,
            word_base_ms=0,
            official_turns=[
                {"speaker_id": 5, "start_ms": 0, "end_ms": 1200},
                {"speaker_id": 2, "start_ms": 1200, "end_ms": 2400},
            ],
            speaker_id_map=labels.labels,
        )
        self.assertEqual(projection["status"], "split_verified")
        self.assertEqual(
            [cue["speaker_id"] for cue in projection["cues"]],
            [1, 2],
        )
        self.assertEqual(
            "".join(cue["text"] for cue in projection["cues"]),
            "Hello world",
        )

    def test_unapproved_channel_is_not_given_a_number(self):
        labels = StableSpeakerChannels()
        labels.resolve(5, 1200)
        projection = project_confirmed_caption(
            text="Hello world",
            words=[
                {"text": "Hello ", "start_ms": 0, "end_ms": 1200},
                {"text": "world", "start_ms": 1200, "end_ms": 2400},
            ],
            caption_start_ms=0,
            caption_end_ms=2400,
            word_base_ms=0,
            official_turns=[
                {"speaker_id": 5, "start_ms": 0, "end_ms": 1200},
                {"speaker_id": 2, "start_ms": 1200, "end_ms": 2400},
            ],
            speaker_id_map=labels.labels,
        )
        self.assertEqual([cue["speaker_id"] for cue in projection["cues"]], [1, -1])

    def test_overlap_audit_uses_identical_display_namespace(self):
        result = overlap_content_risk(
            acoustic_turns=[
                {"speaker_id": 5, "start_ms": 0, "end_ms": 2100},
                {"speaker_id": 2, "start_ms": 500, "end_ms": 1950},
            ],
            caption_turns=[{
                "start_ms": 0, "end_ms": 2100, "speaker_id": 1,
            }],
            speaker_id_map={5: 1, 2: 2},
        )
        self.assertGreater(result["overlap_clock_underrepresented_ms"], 0)
        self.assertEqual(result["review_spans"][0]["speaker_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()

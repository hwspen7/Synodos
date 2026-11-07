"""Optional verified word timing and official Nemotron speaker projection.

All tests are deterministic: no model weights or CUDA are needed.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from speech.captions.quality_guard import verify_utterances
from speech.diarization.nemotron_activity import choose_caption_speaker_from_turns
from tests.quality.speakers.test_nemotron_processor_fusion import (
    FakeAcoustics, processor_for_test,
)


class CaptionProjectionTests(unittest.TestCase):
    def test_word_bounds_keep_original_text_and_legacy_default(self):
        piece = SimpleNamespace(
            start=0.0, end=4.0, text="Hello world",
            words=[
                SimpleNamespace(start=1.2, end=1.7, word="Hello "),
                SimpleNamespace(start=2.0, end=2.5, word="world"),
            ],
        )
        vad = [{"start": 0, "end": 64000}]
        legacy = verify_utterances([piece], vad, 5000, "en")
        improved = verify_utterances(
            [piece], vad, 5000, "en", prefer_word_bounds=True,
        )
        self.assertEqual([x.text for x in improved], ["Hello world"])
        self.assertEqual((legacy[0].start_ms, legacy[0].end_ms), (0, 4000))
        self.assertEqual((improved[0].start_ms, improved[0].end_ms), (1200, 2500))
        self.assertEqual(improved[0].timestamp_quality, "word")
        self.assertEqual(len(improved[0].words), 2)

    def test_official_projection_cannot_select_suppressed_raw_channel(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 0, "end_ms": 200, "speaker_id": 2},
        ]
        decision = choose_caption_speaker_from_turns(turns, 0, 1000)
        self.assertEqual(decision.speaker_id, 1)
        self.assertEqual(decision.status, "official_turn_primary")
        self.assertNotEqual(decision.as_dict()["score_kind"],
                            "calibrated_identity_probability")

    def test_official_projection_abstains_on_equal_overlap(self):
        turns = [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 2},
        ]
        decision = choose_caption_speaker_from_turns(turns, 0, 1000)
        self.assertIsNone(decision.speaker_id)
        self.assertEqual(decision.status, "official_turn_ambiguous")

    def test_final_processor_uses_official_turns_not_raw_activity(self):
        class FinalAcoustic(FakeAcoustics):
            def is_complete(self):
                return True

            def final_turns_snapshot(self):
                return [
                    {"start_ms": 0, "end_ms": 1000, "speaker_id": 3},
                ]

        processor = processor_for_test()
        activity = np.zeros((100, 8), dtype=np.float32)
        activity[:, 6] = 0.99  # Channel 7, suppressed by official model result.
        processor.independent_diarization = FinalAcoustic(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], 1)
        self.assertEqual(processor.snapshot_state()["segments"][0]["speaker_id"], 1)

    def test_official_empty_turns_reject_legacy_fallback(self):
        class FinalNoVoice(FakeAcoustics):
            def is_complete(self):
                return True

            def final_turns_snapshot(self):
                return []

        processor = processor_for_test()
        processor.independent_diarization = FinalNoVoice(
            np.zeros((100, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], -1)
        saved = processor.snapshot_state()["segments"][0]
        self.assertEqual(saved["speaker_id"], -1)
        self.assertEqual(saved["speaker_final_resolution_status"], "speaker_attribution_deferred")


if __name__ == "__main__":
    unittest.main()

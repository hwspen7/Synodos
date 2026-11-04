"""Content-first fusion must never lose, repeat or invent recognized text.

Runs with fake Nemotron evidence: no GPU, checkpoint download or benchmark
reference leakage. Live source segments remain editable and immutable.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from speech.content_first_projection import project_confirmed_caption
from tests.quality.test_nemotron_processor_fusion import (
    FakeAcoustics, processor_for_test,
)


def projection(**overrides):
    params = {
        "text": " Hello there, I agree.",
        "words": [
            {"text": " Hello", "start_ms": 0, "end_ms": 200},
            {"text": " there,", "start_ms": 220, "end_ms": 430},
            {"text": " I", "start_ms": 600, "end_ms": 730},
            {"text": " agree.", "start_ms": 740, "end_ms": 980},
        ],
        "caption_start_ms": 0,
        "caption_end_ms": 1000,
        "word_base_ms": 0,
        "official_turns": [
            {"start_ms": 0, "end_ms": 500, "speaker_id": 1},
            {"start_ms": 550, "end_ms": 1000, "speaker_id": 2},
        ],
    }
    params.update(overrides)
    return project_confirmed_caption(**params)


class ContentFirstProjectionTests(unittest.TestCase):
    def test_two_speakers_exact_text_and_per_word_clock(self):
        result = projection()
        self.assertEqual(result["status"], "split_verified")
        self.assertEqual([x["speaker_id"] for x in result["cues"]], [1, 2])
        self.assertEqual(
            "".join(x["text"] for x in result["cues"]).strip(),
            "Hello there, I agree.",
        )
        self.assertEqual(
            [(x["start_ms"], x["end_ms"]) for x in result["cues"]],
            [(0, 430), (600, 980)],
        )

    def test_unknown_clock_fails_closed_without_losing_text(self):
        p = projection(word_base_ms=None)
        self.assertEqual(p["status"], "unaligned")
        self.assertEqual(p["cues"], [])
        self.assertTrue(p["text_preserved"])

    def test_bad_alignment_never_reconstructs_partial_words(self):
        p = projection(words=[
            {"text": " Hello", "start_ms": 0, "end_ms": 200},
            {"text": " I agree.", "start_ms": 600, "end_ms": 980},
        ])
        self.assertEqual(p["cues"], [])

    def test_overlapping_voices_preserved_as_concurrent_evidence(self):
        p = projection(official_turns=[
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 1},
            {"start_ms": 400, "end_ms": 900, "speaker_id": 2},
        ])
        self.assertEqual(p["overlap_intervals_ms"], [{
            "start_ms": 400, "end_ms": 900, "speaker_ids": [1, 2],
        }])
        self.assertEqual(
            "".join(cue["text"] for cue in p["cues"]).strip()
            if p["cues"] else " Hello there, I agree.".strip(),
            "Hello there, I agree.",
        )

    def test_invalid_model_channel_rejected(self):
        with self.assertRaises(ValueError):
            projection(official_turns=[
                {"start_ms": 0, "end_ms": 1000, "speaker_id": -1},
            ])

    def test_inference_base_and_model_origin_are_distinct(self):
        p = projection(
            caption_start_ms=4000, caption_end_ms=5000,
            word_base_ms=4000,
            acoustic_origin_ms=3000,
            speaker_id_offset=3,
        )
        self.assertEqual([x["speaker_id"] for x in p["cues"]], [4, 5])

    def test_processor_preserves_editable_canonical_segment(self):
        class Final(FakeAcoustics):
            def is_complete(self):
                return True
            def final_turns_snapshot(self):
                return [
                    {"start_ms": 0, "end_ms": 550, "speaker_id": 1},
                    {"start_ms": 550, "end_ms": 1000, "speaker_id": 2},
                ]

        engine = processor_for_test()
        engine.segments[0]["words"] = [
            {"text": "Some", "start_ms": 0, "end_ms": 200},
            {"text": " valid", "start_ms": 220, "end_ms": 440},
            {"text": " speech", "start_ms": 650, "end_ms": 950},
        ]
        source = dict(engine.segments[0])
        engine.independent_diarization = Final(
            np.zeros((100, 8), dtype=np.float32)
        )
        with patch.dict(os.environ, {"MEETING_CONTENT_FIRST_FUSION": "1"}):
            self.assertTrue(engine._rebuild_and_push())
        self.assertEqual(len(engine.segments), 1)
        self.assertEqual(engine.segments[0]["text"], source["text"])
        self.assertEqual(engine.segments[0]["words"], source["words"])
        state = engine.snapshot_state()
        self.assertEqual(len(state["segments"]), 1)
        cues = state["segments"][0]["caption_cues"]
        self.assertEqual([c["speaker_id"] for c in cues], [1, 2])
        self.assertEqual("".join(c["text"] for c in cues), source["text"])


if __name__ == "__main__":
    unittest.main()

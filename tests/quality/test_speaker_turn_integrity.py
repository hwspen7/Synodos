"""Safety properties for asynchronously returned speaker/word partitions."""
from __future__ import annotations

import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from speech.turn_integrity import valid_turn_partition


class SpeakerTurnIntegrityTests(unittest.TestCase):
    def test_original_fallback_is_valid(self):
        self.assertTrue(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world.", "emb": None},
        ]))

    def test_multiple_speaker_turns_keep_every_recognized_character(self):
        self.assertTrue(valid_turn_partition(1000, 4800, "Hello, world. Yes!", [
            {"start": 1000, "end": 2799, "text": "Hello, world."},
            {"start": 2800, "end": 4800, "text": " Yes!"},
        ]))

    def test_truncated_text_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 4800, "Hello world. Yes!", [
            {"start": 1000, "end": 2799, "text": "Hello world."},
        ]))

    def test_modified_punctuation_or_hallucinated_word_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world!"},
        ]))
        self.assertFalse(valid_turn_partition(1000, 3000, "Hello world.", [
            {"start": 1000, "end": 3000, "text": "Hello world. Extra"},
        ]))

    def test_out_of_bounds_or_material_overlap_is_rejected(self):
        self.assertFalse(valid_turn_partition(1000, 4800, "First second", [
            {"start": 1000, "end": 3000, "text": "First"},
            {"start": 2800, "end": 4700, "text": " second"},
        ]))
        self.assertFalse(valid_turn_partition(1000, 4800, "First second", [
            {"start": 1000, "end": 3000, "text": "First"},
            {"start": 3000, "end": 5100, "text": " second"},
        ]))

    def test_small_rounding_tolerance_is_accepted(self):
        self.assertTrue(valid_turn_partition(1000, 4800, "A B", [
            {"start": 990, "end": 3000, "text": "A"},
            {"start": 2980, "end": 4840, "text": " B"},
        ]))

    def test_malformed_worker_output_never_replaces_text(self):
        for candidate in (None, [], [None], ["text"], [
            {"start": True, "end": 2000, "text": "Hi"},
        ], [{"start": 1000, "end": 1000, "text": "Hi"}], [
            {"start": 1000, "end": 2000, "text": ""},
        ]):
            with self.subTest(candidate=candidate):
                self.assertFalse(valid_turn_partition(1000, 2000, "Hi", candidate))

    def test_nfkc_and_spacing_are_only_allowed_normalizations(self):
        self.assertTrue(valid_turn_partition(0, 2000, "\uFF21\uFF22 C", [
            {"start": 0, "end": 2000, "text": "AB C"},
        ]))
        self.assertFalse(valid_turn_partition(0, 2000, "A B C", [
            {"start": 0, "end": 2000, "text": "A B D"},
        ]))


class SessionProcessorTurnIntegrityTests(unittest.TestCase):
    def _processor(self, pieces):
        from sessions.processor import SessionProcessor

        # Use the real asynchronous collection method without starting a
        # thread or loading ASR/diarization models.
        processor = object.__new__(SessionProcessor)
        processor.settings = SimpleNamespace(sample_rate=16000)
        processor.segments = [{
            "id": 1,
            "_audio_base_sample": 0,
            "start_sample": 16000,
            "end_sample": 48000,
            "text": "Hello world.",
            "language": "en",
            "words": [],
            "emb": None,
            "cluster": -1,
            "speaker_uncertain": True,
            "speaker_evidence_status": "pending_turn_analysis",
            "translation": "",
        }]
        processor._next_segment_id = 2
        processor._ecapa_shadow_audio = {}
        processor._acoustic_turn_splits = 0
        processor._last_reclustered_segment_id = 1
        processor._enqueue_translation = lambda segment: None
        future = Future()
        future.set_result(pieces)
        processor._speaker_tasks = {1: future}
        processor._speaker_deferred = {}
        return processor

    def test_rejected_async_worker_preserves_committed_text_and_bounds(self):
        proc = self._processor([
            {"start": 1000, "end": 2000, "text": "Hello", "emb": None},
        ])
        self.assertTrue(proc._collect_speakers())
        self.assertEqual(len(proc.segments), 1)
        self.assertEqual(proc.segments[0]["text"], "Hello world.")
        self.assertEqual(proc.segments[0]["start_sample"], 16000)
        self.assertEqual(proc.segments[0]["end_sample"], 48000)
        self.assertTrue(proc.segments[0]["speaker_uncertain"])
        self.assertEqual(proc.segments[0]["speaker_evidence_status"], "turn_split_rejected")

    def test_valid_async_worker_can_split_without_losing_text(self):
        proc = self._processor([
            {"start": 1000, "end": 2000, "text": "Hello", "emb": None},
            {"start": 2000, "end": 3000, "text": " world.", "emb": None},
        ])
        # The legacy splitter remains available for deterministic audits.
        with patch.dict("os.environ", {"MEETING_CONTENT_FIRST_FUSION": "0"}):
            self.assertTrue(proc._collect_speakers())
        self.assertEqual(len(proc.segments), 2)
        self.assertEqual(
            "".join(x["text"] for x in proc.segments),
            "Hello world.",
        )
        self.assertEqual(proc._acoustic_turn_splits, 1)


if __name__ == "__main__":
    unittest.main()

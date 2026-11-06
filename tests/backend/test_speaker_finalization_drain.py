"""Regression coverage for bounded final speaker drain and live checkpoints."""
from __future__ import annotations

import time
import unittest
from unittest.mock import Mock

from sessions.processor import SessionProcessor


class FinalSpeakerDrainTests(unittest.TestCase):
    @staticmethod
    def processor():
        processor = object.__new__(SessionProcessor)
        processor._speaker_tasks = {}
        processor._speaker_deferred = {}
        return processor

    def test_drain_handles_a_new_wave_submitted_after_first_completion(self):
        processor = self.processor()
        processor._speaker_tasks[1] = object()
        processor._speaker_deferred[2] = object()
        calls = []

        def collect(*, wait_seconds):
            calls.append(wait_seconds)
            if len(calls) == 1:
                processor._speaker_tasks = {2: object()}
                processor._speaker_deferred.clear()
            else:
                processor._speaker_tasks.clear()
            return True

        processor._collect_speakers = Mock(side_effect=collect)
        self.assertTrue(processor._drain_final_speaker_analysis(
            budget_seconds=1.0,
        ))
        self.assertEqual(len(calls), 2)
        self.assertFalse(processor._speaker_tasks)
        self.assertFalse(processor._speaker_deferred)

    def test_deferred_only_wave_is_submitted_and_then_drained(self):
        processor = self.processor()
        processor._speaker_deferred = {2: object()}
        calls = []

        def collect(*, wait_seconds):
            calls.append(wait_seconds)
            if len(calls) == 1:
                processor._speaker_deferred.clear()
                processor._speaker_tasks[2] = object()
            else:
                processor._speaker_tasks.clear()
            return True

        processor._collect_speakers = Mock(side_effect=collect)
        self.assertTrue(processor._drain_final_speaker_analysis(
            budget_seconds=1.0,
        ))
        self.assertEqual(len(calls), 2)

    def test_budget_expires_for_permanently_saturated_deferred_queue(self):
        processor = self.processor()
        processor._speaker_deferred[2] = object()
        processor._collect_speakers = Mock(return_value=False)
        start = time.monotonic()
        self.assertFalse(processor._drain_final_speaker_analysis(
            budget_seconds=.02,
        ))
        self.assertLess(time.monotonic() - start, .3)
        self.assertIn(2, processor._speaker_deferred)

    def test_live_speaker_change_can_autosave_without_new_asr_segment(self):
        processor = self.processor()
        processor.segments = [{"id": 1, "text": "confirmed speech"}]
        processor.meeting_id = "stable"
        processor.meeting_store = object()
        processor._phase07_autosave_interval = 60.0
        processor._phase07_last_autosave = time.monotonic() - 100.0
        processor._phase07_saved_segment_id = 1
        processor._do_save = Mock(return_value=True)
        processor._autosave_if_due(True)
        processor._do_save.assert_called_once_with()
        # The cadence still prevents repeated saves during busy updates.
        processor._phase07_last_autosave = time.monotonic()
        processor._autosave_if_due(True)
        processor._autosave_if_due(False)
        processor._do_save.assert_called_once_with()

    def test_no_save_when_autosave_disabled_or_no_confirmed_segments(self):
        processor = self.processor()
        processor.segments = []
        processor.meeting_id = "stable"
        processor.meeting_store = object()
        processor._phase07_autosave_interval = 60.0
        processor._phase07_last_autosave = time.monotonic() - 100.0
        processor._do_save = Mock(return_value=True)
        processor._autosave_if_due(True)
        processor.segments = [{"id": 1, "text": "confirmed"}]
        processor._phase07_autosave_interval = 0
        processor._autosave_if_due(True)
        processor._do_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""No-GPU realtime WebSocket replay diagnostics and MOPA frame contracts."""
from __future__ import annotations

import unittest

import numpy as np

from api.socket_transport import AudioFrameSequence
from tools.live_caption_ws_replay import capture_event, new_capture, pcm_packet


class LiveCaptionWsReplayTests(unittest.TestCase):
    def test_pcm_packet_matches_actual_audio_protocol_and_rejects_gap(self):
        sequence = AudioFrameSequence()
        first = pcm_packet(np.array([1, -1, 200], dtype=np.int16), 0, 0)
        samples, duplicate = sequence.accept(first)
        self.assertFalse(duplicate)
        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[0], 1 / 32768)
        second = pcm_packet(np.zeros(3200, dtype=np.int16), 1, 3)
        samples, duplicate = sequence.accept(second)
        self.assertEqual(len(samples), 3200)
        self.assertFalse(duplicate)
        _, duplicate = sequence.accept(second)
        self.assertTrue(duplicate)
        with self.assertRaises(ValueError):
            sequence.accept(pcm_packet(np.ones(3, dtype=np.int16), 3, 3203))
        with self.assertRaises(ValueError):
            pcm_packet(np.array([], dtype=np.int16), 0, 0)

    def test_partials_are_revisioned_and_clear_not_a_new_caption(self):
        state = new_capture()
        self.assertFalse(capture_event(state, {
            "type": "partial", "revision": 1, "text": "Hello", "language": "en"
        }, 3200))
        self.assertEqual(state["first_preview_ms"], 3200)
        self.assertFalse(capture_event(state, {
            "type": "partial", "revision": 2, "text": "", "language": "und"
        }, 3600))
        self.assertEqual(state["first_preview_ms"], 3200)
        self.assertEqual(state["clear_events"], 1)
        capture_event(state, {"type": "partial", "revision": 1,
                              "text": "Stale"}, 3800)
        self.assertIn("nonmonotonic_partial_revision", state["flags"])

    def test_full_snapshots_do_not_count_recluster_as_new_text(self):
        state = new_capture()
        capture_event(state, {"type": "transcript", "version": 1,
                              "segments": [
                                  {"id": 1, "speaker_id": -1, "text": "Hi"}
                              ]}, 8500)
        capture_event(state, {"type": "transcript", "version": 2,
                              "segments": [
                                  {"id": 1, "speaker_id": 3, "text": "Hi"}
                              ]}, 9000)
        self.assertEqual(state["first_confirmed_ms"], 8500)
        self.assertEqual(state["final_segment_count"], 1)
        self.assertEqual(state["speaker_reassignments"], 1)
        self.assertEqual(state["events"]["transcript"], 2)

    def test_diagnoses_speaker_changes_without_claiming_identity_errors(self):
        state = new_capture()
        capture_event(state, {
            "type": "transcript", "version": 1, "segments": [
                {"id": 10, "speaker_id": -1,
                 "speaker_name": "Unidentified Speaker", "text": "Hello"},
                {"id": 11, "speaker_id": 1,
                 "speaker_name": "Speaker 1", "text": "Morning"},
            ],
        }, 6000)
        capture_event(state, {
            "type": "transcript", "version": 2, "segments": [
                {"id": 10, "speaker_id": 2,
                 "speaker_name": "Speaker 2", "text": "Hello"},
                {"id": 11, "speaker_id": 3,
                 "speaker_name": "Speaker 3", "text": "Morning"},
            ],
        }, 6400)
        capture_event(state, {
            "type": "transcript", "version": 3, "segments": [
                {"id": 10, "speaker_id": 1,
                 "speaker_name": "Speaker 1", "text": "Hello"},
                {"id": 11, "speaker_id": -1,
                 "speaker_name": "Unidentified Speaker", "text": "Morning revised"},
            ],
        }, 7100)
        self.assertEqual(state["speaker_reassignments"], 4)
        self.assertEqual(dict(state["speaker_change_types"]), {
            "unidentified_to_labeled": 1,
            "labeled_to_different_labeled": 2,
            "labeled_to_unidentified": 1,
        })
        self.assertEqual(len(state["speaker_change_counts"]), 2)
        self.assertEqual(max(state["speaker_change_counts"].values()), 2)
        self.assertEqual(state["confirmed_text_changes"], 1)
        self.assertEqual(len(state["changed_text_ids"]), 1)
        self.assertEqual(state["confirmed_segment_disappearances"], 0)
        # A confirmed segment disappears from a later complete snapshot.
        capture_event(state, {
            "type": "transcript", "version": 4, "segments": [
                {"id": 10, "speaker_id": 1,
                 "speaker_name": "Speaker 1", "text": "Hello"},
            ],
        }, 7400)
        self.assertEqual(state["confirmed_segment_disappearances"], 1)
        self.assertEqual(state["max_visible_segments"], 2)

    def test_warnings_are_grouped_and_report_contains_only_bounded_samples(self):
        state = new_capture()
        for i in range(15):
            capture_event(state, {
                "type": "feature_warning",
                "feature": "diarization",
                "message": "Speaker analysis backlog full " + str(i) * 500,
            }, 1500 + i)
        self.assertEqual(state["events"]["feature_warning"], 15)
        self.assertEqual(state["feature_warning_counts"]["diarization"], 15)
        self.assertEqual(len(state["feature_warning_samples"]), 12)
        self.assertLessEqual(
            max(len(item["message"])
                for item in state["feature_warning_samples"]), 240
        )
        self.assertEqual(state["speaker_reassignments"], 0)

    def test_stop_ack_and_terminal_confirmation(self):
        state = new_capture()
        capture_event(state, {"type": "audio_received",
                              "samples": 16000, "matched": True}, 1200)
        self.assertTrue(state["audio_delivery_matched"])
        self.assertTrue(capture_event(state, {"type": "closed", "saved": True}, 2000))
        self.assertTrue(state["server_closed"])


if __name__ == "__main__":
    unittest.main()

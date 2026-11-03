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

    def test_stop_ack_and_terminal_confirmation(self):
        state = new_capture()
        capture_event(state, {"type": "audio_received",
                              "samples": 16000, "matched": True}, 1200)
        self.assertTrue(state["audio_delivery_matched"])
        self.assertTrue(capture_event(state, {"type": "closed", "saved": True}, 2000))
        self.assertTrue(state["server_closed"])


if __name__ == "__main__":
    unittest.main()

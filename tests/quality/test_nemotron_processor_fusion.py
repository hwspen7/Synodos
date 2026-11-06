"""Integration contracts for published Nemotron/CAM++ speaker projections."""
from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.config import SpeechSettings


class FakeAcoustics:
    def __init__(self, activity):
        self.values = activity

    def activity_snapshot(self):
        return self.values


def processor_for_test():
    settings = Settings(
        asr_model="mock-asr",
        fsmn_vad_model="mock-vad",
        punc_model="mock-punc",
        spk_model="mock-spk",
    )
    # These tests validate speaker projection, not the ASR executable.
    # A user's interactive shell may select hybrid ASR without exporting
    # MEETING_HYBRID_QWEN_PYTHON. Use a fixed mock speech configuration
    # instead of inheriting unrelated machine-specific environment values.
    with patch("sessions.processor.load_settings",
               return_value=SpeechSettings(engine="whisper")):
        processor = SessionProcessor(
            "fusion-case", AudioBuffer(16000, max_seconds=30),
            settings, object(), asyncio.Queue(),
        )
    processor.segments = [{
        "id": 1,
        "_audio_base_sample": 0,
        "start_sample": 0,
        "end_sample": 16000,
        "text": "Some valid speech",
        "words": [],
        "language": "en",
        "emb": np.array([1., 0.], dtype=np.float32),
        "cluster": 0,
        "speaker_id": 17,
        "speaker_name": "Speaker 17",
    }]
    return processor


class LiveFusionTests(unittest.TestCase):
    def test_backpressure_defers_voiceprint_and_retries_after_slot_frees(self):
        from concurrent.futures import Future
        processor = processor_for_test()
        processor.models = SimpleNamespace(
            embed=lambda audio: np.array([1., 0.], dtype=np.float32),
        )
        sentence = {"start": 0, "end": 1000, "text": "Some valid speech",
                    "words": []}
        clip = np.full(16000, 0.1, dtype=np.float32)
        self.assertTrue(processor._defer_speaker_analysis(1, sentence, clip))
        processor.segments[0]["speaker_uncertain"] = True
        processor.segments[0]["speaker_evidence_status"] = "deferred_turn_analysis"
        self.assertEqual(len(processor._speaker_deferred), 1)
        # The first retry remains saturated, the next gets an actual future.
        future = Future()
        with patch("sessions.processor.submit_speaker_analysis",
                   side_effect=[None, future]) as submit:
            processor._retry_deferred_speakers()
            self.assertEqual(len(processor._speaker_deferred), 1)
            self.assertEqual(len(processor._speaker_tasks), 0)
            processor._retry_deferred_speakers()
        self.assertEqual(submit.call_count, 2)
        self.assertEqual(len(processor._speaker_deferred), 0)
        self.assertIs(processor._speaker_tasks[1], future)
        self.assertEqual(
            processor.segments[0]["speaker_evidence_status"],
            "pending_turn_analysis",
        )
        self.assertEqual(processor.segments[0]["text"], "Some valid speech")
        self.assertEqual(
            (processor.segments[0]["start_sample"],
             processor.segments[0]["end_sample"]), (0, 16000),
        )

    def test_backpressure_retry_queue_is_bounded_and_keeps_unknown_honest(self):
        processor = processor_for_test()
        samples = np.ones(16000, dtype=np.float32)
        for sid in range(1, 9):
            self.assertTrue(processor._defer_speaker_analysis(
                sid, {"start": 0, "end": 1000, "text": str(sid)}, samples,
            ))
        self.assertFalse(processor._defer_speaker_analysis(
            9, {"start": 0, "end": 1000, "text": "ninth"}, samples,
        ))
        self.assertEqual(len(processor._speaker_deferred), 8)
        self.assertFalse(processor._defer_speaker_analysis(
            10, {}, np.ones(480001, dtype=np.float32),
        ))

    def test_speaker_fixture_is_independent_of_hybrid_shell_settings(self):
        with patch.dict(os.environ, {
            "MEETING_ASR_ENGINE": "hybrid",
            "MEETING_HYBRID_QWEN_PYTHON": "",
        }):
            processor = processor_for_test()
        self.assertEqual(processor._speech_settings.engine, "whisper")
        # The production speech configuration validator must remain strict.
        with self.assertRaisesRegex(
            ValueError, "Hybrid ASR requires MEETING_HYBRID_QWEN_PYTHON"
        ):
            SpeechSettings(engine="hybrid", hybrid_qwen_python="").validate()

    def test_opt_in_multi_speaker_diagnostics_do_not_rewrite_camplus(self):
        processor = processor_for_test()
        processor.segments.append({
            **processor.segments[0],
            "id": 2, "start_sample": 16000, "end_sample": 32000,
            "text": "Second human reply",
            "emb": np.array([0., 1.], dtype=np.float32),
            "speaker_id": 18, "speaker_name": "Speaker 18",
        })
        activity = np.zeros((200, 8), dtype=np.float32)
        activity[:100, 0] = 0.95
        activity[100:, 1] = 0.95
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        # One uncorroborated second voice is NOT enough to publish Speaker 2.
        self.assertEqual([row["speaker_id"]
                          for row in processor.snapshot_state()["segments"]],
                         [1, -1])
        self.assertEqual([row["speaker_id"] for row in processor.segments],
                         [17, 18])

    def test_four_final_channels_reconcile_missing_and_provisional_captions(self):
        processor = processor_for_test()
        original = processor.segments[0]
        turns = [{"speaker_id": 1, "start_ms": 0, "end_ms": 1000}]
        for channel in range(2, 5):
            start_ms = (channel - 1) * 5000
            turns.append({
                "speaker_id": channel, "start_ms": start_ms, "end_ms": start_ms + 4000,
            })
            for offset in range(2):
                begin = start_ms + offset * 2000
                processor.segments.append({
                    **original, "id": len(processor.segments) + 1,
                    "start_sample": begin * 16, "end_sample": (begin + 2000) * 16,
                    "emb": None if offset == 0 else np.array([.74, .672], dtype=np.float32),
                    "cluster": -3 if offset == 0 else channel,
                    "speaker_id": -1 if offset == 0 else 20 + channel,
                    "speaker_evidence_status": (
                        "multi_turn_acoustic_evidence_only" if offset == 0 else "turn_voiceprint"
                    ),
                })
        processor.independent_diarization = SimpleNamespace(
            activity_snapshot=lambda: None,
            is_complete=lambda: True,
            final_turns_snapshot=lambda: turns,
        )
        self.assertTrue(processor._rebuild_and_push())
        first = processor.snapshot_state()
        processor._rebuild_and_push()
        state = processor.snapshot_state()
        expected = [1, 2, 2, 3, 3, 4, 4]
        self.assertEqual([row["speaker_id"] for row in first["segments"]], expected)
        self.assertEqual([row["speaker_id"] for row in state["segments"]], expected)
        self.assertTrue(all(row["speaker_uncertain"] for row in state["segments"][1:]))
        self.assertTrue(all(row["text"] == original["text"] for row in state["segments"]))

    def test_one_human_stays_speaker_one_across_five_nemotron_channels(self):
        processor = processor_for_test()
        for index in range(2, 6):
            processor.segments.append({
                **processor.segments[0],
                "id": index,
                "start_sample": (index - 1) * 32000,
                "end_sample": index * 32000,
                "text": f"One person's phrase {index}",
                "emb": np.array([1.0, .04 * (index - 1)], dtype=np.float32),
                "speaker_id": 16 + index,
                "speaker_name": f"Speaker {16 + index}",
            })
        processor.segments[0]["end_sample"] = 32000
        activity = np.zeros((1000, 8), dtype=np.float32)
        for channel in range(5):
            activity[channel * 200:(channel + 1) * 200, channel] = .97
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(
            [segment["speaker_id"] for segment in processor.snapshot_state()["segments"]],
            [1, 1, 1, 1, 1],
        )
        self.assertEqual(len(set(processor._nemotron_channels.labels.values())), 1)
        self.assertEqual(len(processor.segments), 5)

    def test_missing_asr_voiceprint_is_recovered_from_matching_acoustic_window(self):
        processor = processor_for_test()
        processor.segments[0]["end_sample"] = 32000
        processor.segments.append({
            **processor.segments[0],
            "id": 2,
            "start_sample": 32000,
            "end_sample": 64000,
            "text": "I am still speaking",
            "emb": None,
            "cluster": -3,
            "speaker_id": -1,
            "speaker_name": "Unidentified Speaker",
            "speaker_evidence_status": "turn_analysis_unavailable",
        })
        self.assertTrue(processor._acoustic_timeline.add(
            32000, 64000, np.array([.99, .10], dtype=np.float32),
        ))
        activity = np.zeros((400, 8), dtype=np.float32)
        activity[:200, 0] = .96
        activity[200:, 1] = .96
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(
            [row["speaker_id"] for row in processor.snapshot_state()["segments"]],
            [1, 1],
        )
        self.assertEqual(
            [row["text"] for row in processor.snapshot_state()["segments"]],
            ["Some valid speech", "I am still speaking"],
        )

    def test_two_genuine_voices_require_multiple_distinct_acoustic_turns(self):
        processor = processor_for_test()
        processor.segments[0]["end_sample"] = 32000
        for index in (2, 3):
            processor.segments.append({
                **processor.segments[0],
                "id": index,
                "start_sample": (index - 1) * 32000,
                "end_sample": index * 32000,
                "text": f"Distinct voice turn {index}",
                "emb": np.array([.02 * index, 1.0], dtype=np.float32),
                "speaker_id": 17 + index,
                "speaker_name": f"Speaker {17 + index}",
            })
        activity = np.zeros((600, 8), dtype=np.float32)
        activity[:200, 0] = .98
        activity[200:600, 1] = .98
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        # The first pass confirms a second voice only after its second turn.
        # A subsequent deterministic rebuild backfills the earlier unknown.
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(
            [row["speaker_id"] for row in processor.snapshot_state()["segments"]],
            [1, 2, 2],
        )
        self.assertEqual(
            [row["text"] for row in processor.snapshot_state()["segments"]],
            ["Some valid speech", "Distinct voice turn 2", "Distinct voice turn 3"],
        )

    def test_independent_activity_corrects_caption_but_not_acoustic_cluster(self):
        processor = processor_for_test()
        activity = np.zeros((100, 8), dtype=np.float32)
        activity[:, 2] = 0.95
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], 1)
        self.assertEqual(processor.segments[0]["speaker_id"], 17)
        self.assertEqual(processor.segments[0]["cluster"], 0)
        state = processor.snapshot_state()
        self.assertEqual(state["segments"][0]["speaker_id"], 1)
        self.assertEqual(
            state["segments"][0]["speaker_diarization_source"], "nemotron3",
        )
        self.assertGreater(state["segments"][0]["speaker_activity_score"], .9)

    def test_manual_override_cannot_be_replaced_by_activity(self):
        processor = processor_for_test()
        processor.segments[0].update({
            "speaker_override": True,
            "speaker_override_id": 8,
            "speaker_override_name": "Manual",
            "saved_nemotron_id": 3,
        })
        activity = np.zeros((100, 8), dtype=np.float32)
        activity[:, 0] = 0.98
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertNotIn(1, processor._independent_published)
        state = processor.snapshot_state()
        self.assertEqual(state["segments"][0]["speaker_id"], 8)

    def test_ambiguous_acoustics_never_publish_extra_camplus_speaker(self):
        processor = processor_for_test()
        processor.segments[0]["end_sample"] = 16000
        processor.segments.append({
            **processor.segments[0],
            "id": 2,
            "start_sample": 16000,
            "end_sample": 24000,
            "speaker_id": 18,
            "speaker_name": "Speaker 18",
            "emb": np.array([0., 1.], dtype=np.float32),
        })
        activity = np.zeros((150, 8), dtype=np.float32)
        activity[:100, 2] = 0.95
        activity[100:, 0] = 0.77
        activity[100:, 1] = 0.75
        processor.independent_diarization = FakeAcoustics(activity)
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published, {1: 1, 2: -1})
        state = processor.snapshot_state()
        self.assertEqual([row["speaker_id"] for row in state["segments"]], [1, -1])
        self.assertEqual(
            state["segments"][1]["speaker_diarization_source"],
            "nemotron3_uncertain",
        )
        self.assertEqual(
            state["segments"][1]["speaker_name"],
            "Unidentified Speaker",
        )

    def test_completed_nemotron_unknown_preserves_supported_camplus_identity(self):
        class CompletedFakeAcoustics(FakeAcoustics):
            def is_complete(self):
                return True

        processor = processor_for_test()
        processor.segments[0].update({
            "speaker_id": 1,
            "speaker_name": "Speaker 1",
            "speaker_evidence_status": "turn_voiceprint",
            "speaker_uncertain": False,
        })
        processor.independent_diarization = CompletedFakeAcoustics(
            np.zeros((100, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        result = processor.snapshot_state()["segments"][0]
        self.assertEqual(result["speaker_id"], 1)
        self.assertEqual(result["speaker_name"], "Speaker 1")
        self.assertEqual(
            result["speaker_diarization_source"],
            "camplus_independent_unresolved",
        )
        self.assertNotIn(1, processor._independent_published)

    def test_completed_nemotron_unknown_does_not_merge_two_camplus_voices(self):
        class CompletedFakeAcoustics(FakeAcoustics):
            def is_complete(self):
                return True

        processor = processor_for_test()
        processor.segments[0].update({
            "speaker_id": 1,
            "speaker_name": "Speaker 1",
            "speaker_evidence_status": "turn_voiceprint",
            "speaker_uncertain": False,
        })
        processor.segments.append({
            **processor.segments[0],
            "id": 2,
            "start_sample": 16000,
            "end_sample": 32000,
            "cluster": 1,
            "speaker_id": 2,
            "speaker_name": "Speaker 2",
            "emb": np.array([0, 1], dtype=np.float32),
        })
        processor.independent_diarization = CompletedFakeAcoustics(
            np.zeros((200, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        output = processor.snapshot_state()["segments"]
        self.assertEqual([s["speaker_id"] for s in output], [1, 2])
        self.assertTrue(all(
            s["speaker_diarization_source"]
                == "camplus_independent_unresolved" for s in output
        ))

    def test_completed_track_missing_tail_does_not_invent_legacy_speaker(self):
        class CompletedFakeAcoustics(FakeAcoustics):
            def is_complete(self):
                return True

        processor = processor_for_test()
        processor.independent_diarization = CompletedFakeAcoustics(
            np.zeros((25, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], -1)
        state = processor.snapshot_state()
        self.assertEqual(state["segments"][0]["speaker_id"], -1)
        self.assertEqual(
            state["segments"][0]["speaker_diarization_source"],
            "nemotron3_uncertain",
        )

    def test_resumed_uncertain_speaker_is_not_reclustered_to_legacy_id(self):
        processor = processor_for_test()
        processor._independent_pcm_origin_samples = 16000
        processor.segments[0]["speaker_diarization_source"] = "nemotron3_uncertain"
        processor.segments[0]["speaker_id"] = 17
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published[1], -1)
        self.assertEqual(processor.snapshot_state()["segments"][0]["speaker_id"], -1)

    def test_unscored_activity_does_not_fabricate_speaker(self):
        processor = processor_for_test()
        processor.segments[0]["speaker_id"] = -1
        processor.segments[0]["speaker_name"] = "Unidentified Speaker"
        processor.independent_diarization = FakeAcoustics(
            np.zeros((25, 8), dtype=np.float32),
        )
        self.assertTrue(processor._rebuild_and_push())
        self.assertEqual(processor._independent_published, {})


if __name__ == "__main__":
    unittest.main()

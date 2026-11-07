"""No-model AMI continuous-window tests, including overlapped real-time windows."""
from __future__ import annotations

import json
import unittest

import numpy as np

from tools.evaluation.ami.ami_continuous_voice_audit import (
    audit_continuous_windows,
    extract_continuous_windows,
    reference_window_classification,
    reference_window_coverage,
)


class FakeCamPlus:
    def __init__(self):
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        mean = float(np.mean(audio))
        vector = np.zeros(64, dtype=np.float32)
        if mean < 0.13:
            vector[0] = 1.0
        elif mean > 0.17:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector


class ContinuousAuditTests(unittest.TestCase):
    def test_overlap_is_not_mislabeled_as_pure(self):
        reference = [
            {"start_ms": 0, "end_ms": 3000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2500, "speaker": "B"},
            {"start_ms": 1500, "end_ms": 1800, "speaker": "B"},
        ]
        counts = reference_window_coverage(reference, 0, 2000)
        self.assertEqual(counts, {"A": 2000, "B": 1000})
        state = reference_window_classification(reference, 0, 2000)
        self.assertEqual(state["reference_context"], "multi_voice")
        self.assertIsNone(state["reference_speaker"])
        self.assertEqual(state["other_max_fraction"], 0.5)

        pure = reference_window_classification(
            reference, 2500, 3000,
        )
        self.assertEqual(pure["reference_context"], "pure")
        self.assertEqual(pure["reference_speaker"], "A")

    def test_grid_has_2s_window_1p5s_stride_without_reference(self):
        audio = np.full(10 * 16000, 0.1, dtype=np.float32)
        model = FakeCamPlus()
        vectors, events = extract_continuous_windows(audio, model)
        self.assertEqual(len(events), 6)
        self.assertEqual(
            [x["start_ms"] for x in events],
            [0, 1500, 3000, 4500, 6000, 7500],
        )
        self.assertEqual([x["end_ms"] for x in events][-1], 9500)
        self.assertEqual(len(vectors), 6)
        self.assertEqual(model.calls, 6)
        self.assertEqual(events[0]["status"], "embedded")
        self.assertFalse(any(
            "reference" in str(event) for event in events
        ))

    def test_silence_is_not_embedded_and_is_never_called(self):
        audio = np.zeros(6 * 16000, dtype=np.float32)
        backend = FakeCamPlus()
        vectors, events = extract_continuous_windows(audio, backend)
        self.assertFalse(vectors)
        self.assertEqual(len(events), 3)
        self.assertTrue(all(x["status"] == "rms_rejected" for x in events))
        self.assertEqual(backend.calls, 0)

    def test_speakers_are_scored_only_post_hoc_on_pure_windows(self):
        audio = np.zeros(10 * 16000, dtype=np.float32)
        audio[:4 * 16000] = 0.10
        audio[4 * 16000:8 * 16000] = 0.20
        model = FakeCamPlus()
        vectors, events = extract_continuous_windows(audio, model)
        reference = [
            {"start_ms": 0, "end_ms": 4000, "speaker": "A"},
            {"start_ms": 4000, "end_ms": 8000, "speaker": "B"},
        ]
        report = audit_continuous_windows(vectors, events, reference)
        self.assertTrue(report["available"])
        self.assertEqual(report["total_windows"], 6)
        self.assertEqual(report["successful_embeddings"], 6)
        self.assertEqual(report["pure_voiceprint_counts"]["A"], 2)
        self.assertEqual(report["pure_voiceprint_counts"]["B"], 2)
        self.assertEqual(
            report["reference_context_window_counts"]["multi_voice"], 1
        )
        self.assertEqual(report["raw_cluster_count"], 2)
        self.assertEqual(
            report["pure_window_sample_scores"]["spherical"]["pair_f1"],
            1.0,
        )
        self.assertEqual(
            report["pure_window_sample_scores"]["provisional"]["pair_f1"],
            1.0,
        )
        encoded = json.dumps(report)
        self.assertNotIn('"embedding":', encoded)
        self.assertNotIn('"audio":', encoded)
        self.assertNotIn('"text":', encoded)

    def test_unrepresented_third_speaker_is_not_fabricated(self):
        vectors = [
            np.eye(64, dtype=np.float32)[0] for _ in range(3)
        ] + [np.eye(64, dtype=np.float32)[1] for _ in range(3)]
        events = [
            {
                "start_ms": i * 1500,
                "end_ms": i * 1500 + 2000,
                "status": "embedded", "vector_index": i,
            }
            for i in range(6)
        ]
        reference = [
            {"start_ms": 0, "end_ms": 2500, "speaker": "A"},
            {"start_ms": 3000, "end_ms": 5500, "speaker": "B"},
            {"start_ms": 6000, "end_ms": 8500, "speaker": "C"},
        ]
        report = audit_continuous_windows(vectors, events, reference)
        self.assertEqual(report["true_speakers_from_rttm"], 3)
        self.assertEqual(report["raw_cluster_count"], 2)
        # Artificial voiceprints do not prove the third annotated person
        # is discovered, even if all windows generated vectors.
        self.assertIsInstance(
            report["pure_voiceprint_counts"]["C"], int,
        )

    def test_invalid_inputs_and_empty_embeddings_fail_closed(self):
        model = FakeCamPlus()
        with self.assertRaises(ValueError):
            extract_continuous_windows(np.zeros(100), model)
        with self.assertRaises(ValueError):
            extract_continuous_windows(
                np.array([float("nan")] * 32000), model
            )
        result = audit_continuous_windows(
            [], [{"start_ms": 0, "end_ms": 2000, "status": "rms_rejected"}],
            [{"start_ms": 0, "end_ms": 1000, "speaker": "A"}],
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "insufficient_or_over_budget_embeddings")


if __name__ == "__main__":
    unittest.main()

"""Multi-speaker overlap detection must not treat every speech turn as overlap."""
from __future__ import annotations

import unittest

import numpy as np

from speech.overlap import OverlapDetector, OverlapUnavailable


class Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Timeline:
    def __init__(self, spans):
        self.spans = spans

    def support(self):
        return self.spans

    def __iter__(self):
        return iter(self.spans)


class Annotation:
    def __init__(self):
        self.calls = []

    def get_timeline(self):
        raise AssertionError("Speech support is not an overlap label")

    def get_overlap(self):
        self.calls.append("overlap")
        return Timeline([Span(1.2, 2.6), Span(4.5, 5.2)])


class Output:
    def __init__(self, annotation):
        self.speaker_diarization = annotation


class OverlapTests(unittest.TestCase):
    def test_diarization_wrapper_returns_only_true_overlaps(self):
        annotation = Annotation()
        detector = OverlapDetector(
            pipeline=lambda _: Output(annotation)
        )
        result = detector.detect(np.zeros(16000 * 8, dtype=np.float32))
        self.assertEqual(result, [
            {"start_ms": 1200, "end_ms": 2600,
             "type": "overlapping_speech"},
            {"start_ms": 4500, "end_ms": 5200,
             "type": "overlapping_speech"},
        ])
        self.assertEqual(annotation.calls, ["overlap"])

    def test_legacy_annotation_supports_actual_overlap(self):
        annotation = Annotation()
        result = OverlapDetector(lambda _: annotation).detect(
            np.zeros(16000 * 8, dtype=np.float32),
        )
        self.assertEqual(len(result), 2)

    def test_no_pipeline_and_unsupported_output_fail_closed(self):
        with self.assertRaises(OverlapUnavailable):
            OverlapDetector().detect(np.zeros(16000 * 2))
        with self.assertRaises(OverlapUnavailable):
            OverlapDetector(lambda _: object()).detect(
                np.zeros(16000 * 2, dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()

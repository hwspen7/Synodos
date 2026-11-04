"""Lossless, bounded second-pass speech recovery without Whisper downloads."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from speech.contracts import Utterance
from speech.gap_recovery import (
    recover_missing_voice, uncovered_voice_intervals,
)
from speech.quality_guard import verify_utterances


def vad(*spans):
    return [{"start": a * 16, "end": b * 16} for a, b in spans]


class FakeText:
    def __init__(self, text, start, end, words=None):
        self.text, self.start, self.end = text, start, end
        self.words = words
        self.no_speech_prob = 0.01
        self.avg_logprob = -0.15
        self.compression_ratio = 1.1


class GapRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.ones(10 * 16000, dtype=np.float32) * 0.04

    def test_no_recovery_when_vad_is_covered_by_existing_text(self):
        source = (Utterance("Primary confirmed words", 0, 9000),)
        missing = uncovered_voice_intervals(
            vad((500, 8800)), source, 10000,
        )
        self.assertEqual(missing, ())
        calls = []
        output, stats = recover_missing_voice(
            self.audio, vad((500, 8800)), source,
            lambda x, **kw: calls.append(kw), {}, language="en",
        )
        self.assertEqual(output, source)
        self.assertEqual(calls, [])
        self.assertEqual(stats["recovery_attempted"], 0)

    def test_recovers_missing_early_speech_without_erasing_original(self):
        original = (Utterance("Later words", 5500, 7600),)
        callbacks = []

        def transcribe(chunk, **kwargs):
            callbacks.append(kwargs)
            return [
                FakeText("Recovered actual speech", 0.25, 2.25),
            ], SimpleNamespace(language="en")

        result, stats = recover_missing_voice(
            self.audio,
            vad((0, 2700), (5500, 7500)),
            original,
            transcribe,
            {"vad_filter": True, "word_timestamps": True},
            language="en",
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Recovered actual speech")
        self.assertEqual(result[0].recognition_origin, "gap_recovery")
        self.assertEqual(result[1], original[0])
        self.assertEqual(stats["recovery_confirmed"], 1)
        self.assertEqual(stats["recovery_failed"], 0)
        self.assertTrue(callbacks)
        self.assertTrue(all(x["vad_filter"] is False for x in callbacks))

    def test_retry_failure_cannot_destroy_verified_transcript(self):
        original = (Utterance("Keep this text", 5000, 7500),)
        def fail(chunk, **kwargs):
            raise RuntimeError("GPU unavailable")

        output, stats = recover_missing_voice(
            self.audio, vad((0, 2600), (5000, 7500)),
            original, fail, {}, language="auto",
        )
        self.assertEqual(output, original)
        self.assertEqual(stats["recovery_attempted"], 1)
        self.assertEqual(stats["recovery_failed"], 1)

    def test_overlapping_new_text_is_never_duplicated(self):
        original = (Utterance("Existing", 1500, 3900),)
        def transcribe(chunk, **kwargs):
            return [FakeText("Duplicate", 1.2, 3.7)], SimpleNamespace(
                language="en",
            )
        result, stats = recover_missing_voice(
            self.audio, vad((0, 4000)), original,
            transcribe, {}, language="en",
        )
        self.assertEqual(result, original)
        self.assertEqual(stats["recovery_confirmed"], 0)

    def test_never_fabricates_words_from_silence_or_weak_voice(self):
        untouched = (Utterance("Strong evidence", 8000, 9800),)
        results, stats = recover_missing_voice(
            self.audio, vad((0, 4000)), untouched,
            lambda a, **kw: (
                [FakeText("Imagined", 0.0, 0.1)],
                SimpleNamespace(language="en"),
            ),
            {}, language="en",
        )
        self.assertEqual(results, untouched)
        self.assertEqual(stats["recovery_confirmed"], 0)
        self.assertEqual(
            uncovered_voice_intervals(vad((0, 600)), (), 10000),
            (),
        )

    def test_budget_is_limited_even_on_many_uncovered_regions(self):
        attempted = []
        def transcribe(chunk, **kwargs):
            attempted.append(len(chunk))
            return [], SimpleNamespace(language="en")
        regions = vad(
            (0, 1500), (2000, 3500), (4000, 5500),
            (6000, 7500), (8000, 9500),
        )
        out, metrics = recover_missing_voice(
            self.audio, regions, (), transcribe, {},
            language="en",
        )
        self.assertEqual(out, ())
        self.assertEqual(metrics["recovery_attempted"], 2)
        self.assertEqual(len(attempted), 2)


if __name__ == "__main__":
    unittest.main()

"""Acoustic window segmentation and per-window language correctness."""
import unittest
from types import SimpleNamespace
from speech.audio_windows import local_speech_regions, speech_windows
from speech.multilingual_windows import recognize_windows


class AudioWindowTests(unittest.TestCase):
    def test_different_speech_intervals_remain_distinct(self):
        sr = 16000
        output = speech_windows([{"start": 2*sr, "end": 4*sr},
                                 {"start": 7*sr, "end": 9*sr}], 10*sr)
        self.assertEqual(len(output), 2)
        self.assertLess(output[0][1], output[1][0])

    def test_near_speech_intervals_merge(self):
        sr = 16000
        output = speech_windows([{"start": 2*sr, "end": 3*sr},
                                 {"start": 3*sr+1000, "end": 4*sr}], 6*sr)
        self.assertEqual(len(output), 1)

    def test_long_voiced_interval_bounded(self):
        sr = 16000
        output = speech_windows([{"start": 0, "end": 70*sr}], 70*sr)
        self.assertGreaterEqual(len(output), 4)
        self.assertTrue(all(b-a <= 18*sr for a, b in output))
        self.assertEqual(output[0][0], 0)
        self.assertEqual(output[-1][1], 70*sr)

    def test_invalid_region_ignored(self):
        self.assertEqual(speech_windows([{"start": 9, "end": 3}], 160), [])

    def test_local_region_offsets(self):
        self.assertEqual(local_speech_regions([{"start": 100, "end": 600}], 200, 500),
                         [{"start": 0, "end": 300}])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            speech_windows([], 100, max_window_s=1)


class LanguageWindowTests(unittest.TestCase):
    def _run(self, languages):
        sr = 16000
        regions = [{"start": 1*sr, "end": 3*sr},
                   {"start": 6*sr, "end": 8*sr}]
        count = [0]
        def mocked_asr(samples, **kwargs):
            i = count[0]
            count[0] += 1
            info = SimpleNamespace(language=languages[i][0], language_probability=languages[i][1])
            segment = SimpleNamespace(text="Test words", start=0.1,
                                      end=(len(samples)/sr)-0.1,
                                      no_speech_prob=0.01, avg_logprob=-0.01,
                                      compression_ratio=1.1, words=[])
            return iter([segment]), info
        result = recognize_windows([0.1]*(10*sr), regions, mocked_asr,
                                   {"language": None, "multilingual": False})
        return result, count[0]

    def test_multilingual_labels_not_global(self):
        result, n = self._run([("zh", .95), ("en", .91)])
        self.assertEqual(n, 2)
        self.assertEqual(result.language, "mul")
        self.assertEqual([x.language for x in result.utterances], ["zh", "en"])
        self.assertGreater(result.utterances[1].start_ms, result.utterances[0].start_ms)

    def test_low_confidence_does_not_invent_language(self):
        result, _ = self._run([("pt", .21), ("en", .94)])
        self.assertEqual([x.language for x in result.utterances], ["und", "en"])
        self.assertAlmostEqual(result.utterances[0].language_confidence, .21)

    def test_no_speech(self):
        result = recognize_windows([0.0]*16000, [], lambda *_a, **_k: None, {})
        self.assertEqual(result.utterances, ())

    def test_cap_expensive_windows(self):
        with self.assertRaises(RuntimeError):
            recognize_windows([0.1]*64000, [{"start": 0, "end": 64000}],
                              lambda *_a, **_k: None, {}, max_windows=0)

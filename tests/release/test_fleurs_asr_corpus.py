"""Safety tests: FLEURS selection is data-agnostic, deterministic and private."""
import unittest
from unittest.mock import patch
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import wave
from tools.freeze_fleurs_asr_corpus import choose_indices, encode_pcm16, LANGUAGES, SCHEMA


class FleursCorpusTests(unittest.TestCase):
    def test_exact_target_language_set(self):
        self.assertEqual(set(LANGUAGES), {"en","zh","fr","es","ar","id","ja","ru","de","hi"})
        self.assertEqual(SCHEMA, "meeting-asr-selection-corpus-v1")

    def test_freeze_is_independent_of_transcriptions_and_model_results(self):
        ids = [{"id": i} for i in range(40)]
        a = choose_indices(ids, language="zh", count=10, seed="fixed")
        b = choose_indices(ids, language="zh", count=10, seed="fixed")
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 10)
        self.assertNotEqual(a, choose_indices(ids, language="en", count=10, seed="fixed"))

    def test_fleurs_accepts_actual_hub_commit_shas(self):
        from tools.freeze_fleurs_asr_corpus import SHA
        self.assertIsNotNone(SHA.fullmatch("a" * 40))
        self.assertIsNotNone(SHA.fullmatch("a" * 64))
        self.assertIsNone(SHA.fullmatch("a" * 39))

    def test_pcm_export_is_mono_and_16000(self):
        wav, duration = encode_pcm16({"array": np.zeros(16000, np.float32),
                                      "sampling_rate":16000})
        self.assertEqual(duration, 1)
        with wave.open(io.BytesIO(wav)) as f:
            self.assertEqual((f.getframerate(), f.getnchannels(), f.getsampwidth()),
                             (16000, 1, 2))
            self.assertEqual(f.getnframes(), 16000)

    def test_refuse_nonfinite_audio_and_incompatible_rates(self):
        with self.assertRaises(ValueError):
            encode_pcm16({"array": np.array([float("nan")] * 16000),
                          "sampling_rate": 16000})
        with self.assertRaises(ValueError):
            encode_pcm16({"array": np.zeros(8000), "sampling_rate": 8000})


if __name__ == "__main__":
    unittest.main()

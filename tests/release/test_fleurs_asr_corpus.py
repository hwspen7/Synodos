"""Safety tests: FLEURS selection is data-agnostic, deterministic and private."""
import unittest
from unittest.mock import patch
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import wave
from tools.freeze_fleurs_asr_corpus import choose_indices, choose_streaming_examples, freeze_plan, encode_pcm16, decode_audio_without_torchcodec, LANGUAGES, SCHEMA


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

    def test_streaming_selection_matches_offline_hash_ranking(self):
        records = [{"id": i, "audio": {"bytes": b"not decoded"}} for i in range(40)]
        chosen = choose_streaming_examples(iter(records),
                                           language="en", count=11, seed="frozen")
        indices = choose_indices(records, language="en", count=11, seed="frozen")
        self.assertEqual([row["id"] for row in chosen], indices)
        self.assertTrue(all(row["audio"]["bytes"] == b"not decoded" for row in chosen))

    def test_streaming_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            choose_streaming_examples(iter([{"id": 7}, {"id": 7}]),
                                      language="en", count=1, seed="v1")

    def test_partial_resume_pins_dataset_revision_and_seed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha, path = freeze_plan(root, "validation", 20, "frozen", "a" * 40)
            self.assertEqual(sha, "a" * 40)
            self.assertTrue(path.is_file())
            self.assertEqual(
                freeze_plan(root, "validation", 20, "frozen", None)[0], sha
            )
            with self.assertRaises(ValueError):
                freeze_plan(root, "validation", 20, "changed", None)
            with self.assertRaises(ValueError):
                freeze_plan(root, "validation", 20, "frozen", "b" * 40)

    def test_completed_freeze_is_immutable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze_plan(root, "validation", 20, "frozen", "a" * 40)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                freeze_plan(root, "validation", 20, "frozen", None)

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

    def test_decode_embedded_pcm_wav_without_torchcodec(self):
        array = np.linspace(-.1, .1, 16000, dtype=np.float32)
        encoded, _ = encode_pcm16({"array": array, "sampling_rate": 16000})
        restored = decode_audio_without_torchcodec({"bytes": encoded, "path": None})
        self.assertEqual(restored["sampling_rate"], 16000)
        self.assertEqual(len(restored["array"]), 16000)
        self.assertAlmostEqual(restored["array"][1000], array[1000], delta=.0001)

    def test_decode_invalid_bytes_fails_without_torch_upgrades(self):
        with self.assertRaises(ValueError):
            decode_audio_without_torchcodec({"bytes": b"not recorded audio"})

    def test_refuse_nonfinite_audio_and_incompatible_rates(self):
        with self.assertRaises(ValueError):
            encode_pcm16({"array": np.array([float("nan")] * 16000),
                          "sampling_rate": 16000})
        with self.assertRaises(ValueError):
            encode_pcm16({"array": np.zeros(8000), "sampling_rate": 8000})


if __name__ == "__main__":
    unittest.main()

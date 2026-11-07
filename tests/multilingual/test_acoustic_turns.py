"""Acoustic turn partitioning must preserve words and avoid invented identities."""
import unittest

import numpy as np

from speech.captions.turn_split import split_by_acoustic_turns


class AcousticTurnTests(unittest.TestCase):
    @staticmethod
    def source():
        return {
            "start": 0, "end": 8000, "text": "Alpha beta gamma delta",
            "words": [
                {"start_ms": 0, "end_ms": 1800, "text": "Alpha"},
                {"start_ms": 2000, "end_ms": 3800, "text": "beta"},
                {"start_ms": 4000, "end_ms": 5800, "text": "gamma"},
                {"start_ms": 6000, "end_ms": 7800, "text": "delta"},
            ],
        }

    def test_consistent_voice_remains_one_segment(self):
        audio = np.ones(128000, dtype=np.float32)
        chunks = split_by_acoustic_turns(self.source(), audio, lambda wave: [1.0, 0.0])
        self.assertEqual(len(chunks), 1)
        self.assertIsNotNone(chunks[0]["emb"])

    def test_acoustic_change_separates_word_groups(self):
        audio = np.concatenate((np.ones(64000, dtype=np.float32),
                                -np.ones(64000, dtype=np.float32)))
        def voice(wave):
            return [1.0, 0.0] if float(np.mean(wave)) >= 0 else [0.0, 1.0]
        chunks = split_by_acoustic_turns(self.source(), audio, voice)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(" ".join(chunk["text"] for chunk in chunks),
                         "Alpha beta gamma delta")
        self.assertTrue(all(chunk["end"] > chunk["start"] for chunk in chunks))

    def test_mixed_voice_without_word_alignment_is_unidentified(self):
        audio = np.concatenate((np.ones(64000, dtype=np.float32),
                                -np.ones(64000, dtype=np.float32)))
        sentence = {"start": 0, "end": 8000, "text": "Some words"}
        chunks = split_by_acoustic_turns(sentence, audio,
                 lambda wave: [1.0, 0.0] if float(np.mean(wave)) >= 0 else [0.0, 1.0])
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["ambiguous"])
        self.assertIsNone(chunks[0]["emb"])

    def test_invalid_embedding_never_discards_text(self):
        sentence = self.source()
        audio = np.ones(128000, dtype=np.float32)
        chunks = split_by_acoustic_turns(sentence, audio, lambda wave: None)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], sentence["text"])

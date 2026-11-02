import unittest
import numpy as np
from storage.json_codec import json_safe


class CodecTests(unittest.TestCase):
    def test_numpy_float_vectors_json(self):
        vector = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        state = json_safe({"segments": [{"emb": vector, "text": "recognized"}]})
        self.assertEqual(len(state["segments"][0]["emb"]), 3)
        self.assertEqual(state["segments"][0]["text"], "recognized")

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            json_safe({"emb": float("nan")})

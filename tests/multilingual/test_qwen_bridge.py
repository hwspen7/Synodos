"""Qwen worker serialization tests without importing or loading CUDA."""
import unittest
from unittest.mock import Mock
import numpy as np
from speech.contracts import Recognition, Utterance, Word
from speech.qwen_bridge import decode_recognition
from speech.qwen_worker import encode_recognition, process_request

class QwenLocalBridgeTests(unittest.TestCase):
    def test_wire_round_trip_preserves_text_and_word_times(self):
        original = Recognition(
            utterances=(Utterance(
                "world", 100, 800, "en",
                words=(Word("world", 100, 800),), timestamp_quality="word",
            ),), language="en", engine="qwen3", speech_detected=True)
        recovered = decode_recognition(encode_recognition(original))
        self.assertEqual(original, recovered)

    def test_prewarm_command_does_not_need_gpu_to_test(self):
        backend = Mock()
        self.assertEqual(process_request(backend, {"command": "prewarm"}),
                         {"ready": True})
        backend._get_model.assert_called_once()

    def test_malformed_audio_is_rejected(self):
        backend = Mock()
        with self.assertRaises(ValueError):
            process_request(backend, {
                "command": "recognize",
                "audio_f32le": "Zm9v",
            })
        backend.recognize.assert_not_called()

if __name__ == "__main__":
    unittest.main()

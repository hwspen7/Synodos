"""Qwen worker serialization tests without importing or loading CUDA."""
import io
import json
import sys
from queue import Queue
import unittest
from unittest.mock import Mock
import numpy as np
from speech.contracts import Recognition, Utterance, Word
from speech.qwen_bridge import (
    PREFIX_BYTES, QwenWorkerClient, decode_recognition,
    parse_worker_stdout,
)
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

    def test_non_utf8_windows_native_output_is_ignored(self):
        self.assertIsNone(parse_worker_stdout(
            b"Native log: \xa8\x8f\xff\r\n"
        ))

    def test_protocol_survives_non_utf8_prefix_in_same_line(self):
        wire = PREFIX_BYTES + json.dumps({
            "id": 41, "ok": True, "result": {"ready": True}
        }, separators=(",", ":")).encode("ascii") + b"\n"
        parsed = parse_worker_stdout(b"untrusted \xa8\xff " + wire)
        self.assertEqual(parsed["id"], 41)
        self.assertTrue(parsed["result"]["ready"])

    def test_broken_protocol_is_explicit_failure(self):
        corrupt = PREFIX_BYTES + b'{"id":41,"ok":true,"text":"\xa8"}\n'
        parsed = parse_worker_stdout(corrupt)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["error_type"], "InvalidFrameEncoding")

    def test_raw_reader_survives_bad_bytes_and_closes_queue(self):
        reader = QwenWorkerClient(sys.executable)
        channel = Queue()
        stream = io.BytesIO(
            b"CUDA output: \xa8\xff\n" +
            PREFIX_BYTES + b'{"id":3,"ok":true,"result":{"ready":true}}\n'
        )
        reader._read_output(stream, channel)
        self.assertEqual(channel.get_nowait()["id"], 3)
        self.assertIsNone(channel.get_nowait())
        reader.close()

    def test_old_worker_channel_cannot_poison_new_channel(self):
        reader = QwenWorkerClient(sys.executable)
        old_channel = Queue()
        reader.queue = Queue()
        reader._read_output(io.BytesIO(b""), old_channel)
        self.assertIsNone(old_channel.get_nowait())
        self.assertTrue(reader.queue.empty())
        reader.close()

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

"""Reject malformed browser audio chunks without crashing the socket handler."""

from __future__ import annotations

import unittest

import numpy as np

from api.socket_transport import MAX_PCM_FRAME_BYTES, decode_pcm_frame


class PcmFrameTests(unittest.TestCase):
    def test_valid_signed_16bit_little_endian_samples(self):
        original = np.array([0, -32768, 32767], dtype="<i2")
        output = decode_pcm_frame(original.tobytes())
        self.assertEqual(len(output), 3)
        self.assertAlmostEqual(float(output[1]), -1.0)

    def test_odd_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even"):
            decode_pcm_frame(b"123")

    def test_large_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            decode_pcm_frame(b"0" * (MAX_PCM_FRAME_BYTES + 2))

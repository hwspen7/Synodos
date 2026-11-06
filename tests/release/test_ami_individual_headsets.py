"""Offline tests for pinned AMI individual-headset acquisition integrity."""
import tempfile
import unittest
from pathlib import Path
import wave

from tools.evaluation.ami.prepare_ami_individual_headsets import (
    OFFICIAL_PREFIX, official_url, verify_channel,
)


class IndividualHeadsetSafetyTests(unittest.TestCase):
    def test_urls_only_use_approved_ami_meetings(self):
        url = official_url("EN2002a", 0)
        self.assertEqual(url, OFFICIAL_PREFIX + "/EN2002a/audio/EN2002a.Headset-0.wav")
        with self.assertRaises(ValueError):
            official_url("ZZ9999a", 1)
        with self.assertRaises(ValueError):
            official_url("EN2002a", 9)

    def test_real_pcm_source_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.wav"
            with wave.open(str(path), "wb") as file:
                file.setframerate(16000)
                file.setnchannels(1)
                file.setsampwidth(2)
                file.writeframes(b"\x00\x00" * 32000)
            result = verify_channel(path, 2)
            self.assertEqual(result["duration_seconds"], 2)
            self.assertEqual(len(result["sha256"]), 64)
            with self.assertRaises(ValueError):
                verify_channel(path, 400)
            with self.assertRaises(ValueError):
                verify_channel(Path(temp) / "missing.wav", 2)


if __name__ == "__main__":
    unittest.main()

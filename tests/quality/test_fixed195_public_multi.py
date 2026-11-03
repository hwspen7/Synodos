"""No-gate public multi corpus contracts: authentic stereo, no fake labels."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from tools.fixed195_public_multi import (
    REPO, TARGETS, SCHEMA, select_files, convert_two_channels,
)


class PublicMultiContracts(unittest.TestCase):
    def test_source_is_public_and_has_non_oracle_audio_only_contract(self):
        self.assertEqual(REPO, "datahiveai/multilingual-conversations-sample")
        self.assertEqual(TARGETS, {"en": 10, "hi": 5, "id": 5, "ru": 5})
        self.assertEqual(SCHEMA, "meeting-fixed195-public-dialogue-audio-v1")
        files = [
            f"preview/{lang}_{i:06d}.mp3"
            for lang, total in TARGETS.items()
            for i in range(1, total + 2)
        ]
        result = select_files(files)
        self.assertEqual(sum(map(len, result.values())), 25)
        self.assertEqual(result["en"][0], "preview/en_000001.mp3")
        self.assertEqual(result["en"][-1], "preview/en_000010.mp3")
        with self.assertRaisesRegex(ValueError, "need 5"):
            select_files(["preview/hi_000001.mp3"], {"hi": 5})
        with self.assertRaises(ValueError):
            select_files(["preview/en_000001.mp3"], {"en": 2})

    def test_stereo_tracks_are_preserved_and_mixed_with_bounded_gain(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "real_stereo.wav"
            sr = 16000
            duration = 95
            time = np.arange(duration * sr, dtype=np.float32) / sr
            # Different signals emulate separate authentic microphone tracks.
            left = .08 * np.sin(2 * np.pi * 150 * time)
            right = .06 * np.sin(2 * np.pi * 260 * time)
            sf.write(src, np.stack((left, right), axis=1), sr)
            target = base / "real_clip.wav"
            data = convert_two_channels(src, target)
            self.assertEqual(data["duration_s"], 75.0)
            self.assertEqual(len(data["isolated_audio"]), 2)
            self.assertTrue(target.exists())
            for part in data["isolated_audio"]:
                self.assertTrue((base / part["audio"]).exists())
            self.assertGreater(data["speaker_activity_fractions"][0], .95)
            self.assertGreater(data["speaker_activity_fractions"][1], .95)

    def test_mono_speech_cannot_masquerade_as_dual_speaker(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "fake_mono.wav"
            sf.write(src, np.zeros(95 * 16000, dtype=np.float32), 16000)
            with self.assertRaisesRegex(ValueError, "independent stereo"):
                convert_two_channels(src, base / "out.wav")


if __name__ == "__main__":
    unittest.main()

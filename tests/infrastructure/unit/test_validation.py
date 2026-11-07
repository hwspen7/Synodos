import unittest
from core.config import Settings
from core.validation import validate_application_settings


class ValidationTests(unittest.TestCase):
    def test_default_audio_settings_valid(self):
        validate_application_settings(Settings())

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            validate_application_settings(Settings(vad_threshold_s=-2))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            validate_application_settings(Settings(spk_match_threshold=float("nan")))

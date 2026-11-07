import unittest
from speech.config import load_settings
from unittest.mock import patch

class CompatibilityTests(unittest.TestCase):
    def test_env_validation(self):
        with patch.dict('os.environ', {'MEETING_ASR_ENGINE': 'bad'}):
            with self.assertRaises(ValueError):
                load_settings()

    def test_flag_validation(self):
        with patch.dict('os.environ', {'MEETING_RECORD_AUDIO': 'maybe'}):
            with self.assertRaises(ValueError):
                load_settings()

    def test_native_model_asset_validation(self):
        from speech.asr.streaming import NativeOnlineRecognizer
        with self.assertRaises(ValueError):
            NativeOnlineRecognizer('', '', '', '')

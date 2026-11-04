import unittest

from speech.config import SpeechSettings
from speech.language import SUPPORTED_LANGUAGES, normalize_supported_language


class SupportedLanguageTests(unittest.TestCase):
    def test_product_language_set_has_exactly_twelve_codes(self):
        self.assertEqual(len(SUPPORTED_LANGUAGES), 12)
        self.assertEqual(
            SUPPORTED_LANGUAGES,
            {"en", "zh", "hi", "es", "fr", "ar",
             "bn", "pt", "ru", "id", "de", "ja"},
        )

    def test_auto_and_regional_codes_normalize(self):
        self.assertEqual(normalize_supported_language("auto", allow_auto=True), "auto")
        self.assertEqual(normalize_supported_language("zh-CN"), "zh")
        self.assertEqual(normalize_supported_language("en_US"), "en")
        self.assertIsNone(normalize_supported_language("ko"))
        self.assertIsNone(normalize_supported_language("mul"))

    def test_internal_language_evaluation_remains_available(self):
        self.assertEqual(SpeechSettings(language="ko").validate().language, "ko")

    def test_each_product_language_is_accepted(self):
        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                self.assertEqual(normalize_supported_language(language), language)


if __name__ == "__main__":
    unittest.main()

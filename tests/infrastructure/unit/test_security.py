import unittest
from core.infrastructure_settings import load_infrastructure_settings
from core.security import origin_allowed


class SecurityTests(unittest.TestCase):
    def test_default_browser_origins_and_untrusted_origin(self):
        config = load_infrastructure_settings({})
        self.assertTrue(origin_allowed("http://localhost:5173", config))
        self.assertTrue(origin_allowed("http://127.0.0.1:5173", config))
        self.assertFalse(origin_allowed("http://evil.test:5173", config))
        self.assertFalse(origin_allowed(None, config))
        self.assertFalse(origin_allowed("http://localhost:5173/path", config))

    def test_non_browser_flag_is_explicit(self):
        config = load_infrastructure_settings({"APP_ALLOW_NO_ORIGIN_WS": "true"})
        self.assertTrue(origin_allowed(None, config))

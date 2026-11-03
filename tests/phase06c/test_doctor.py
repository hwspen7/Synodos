"""Doctor must not leak database credentials."""
import unittest
from tools.doctor_06c import describe


class EnvironmentDoctorTests(unittest.TestCase):
    def test_secret_values_not_exposed(self):
        result = describe({"DATABASE_URL": ("postgresql://" + "meeting:" + "TOPSECRET" + "@localhost/db"),
                           "MEETING_AI_API_KEY": "SECRET-API-KEY", "MEETING_ASR_LANGUAGE": "auto"})
        self.assertEqual(result["DATABASE_URL"], "configured")
        self.assertEqual(result["MEETING_AI_API_KEY"], "configured")
        self.assertNotIn("TOPSECRET", repr(result))
        self.assertNotIn("SECRET-API-KEY", repr(result))

    def test_no_db_required(self):
        result = describe({})
        self.assertEqual(result["DATABASE_URL"], "not configured")
        self.assertEqual(result["MEETING_ASR_LANGUAGE"], "<not set>")

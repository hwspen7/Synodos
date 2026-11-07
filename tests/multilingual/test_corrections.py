import unittest
from speech.captions.corrections import correct_segment, RevisionConflict

class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.state = {"segments": [{"id": 4, "text": "old text", "start_sample": 0, "end_sample": 16000}]}

    def test_revision_and_history(self):
        record = correct_segment(self.state, 4, "new text", 0)
        self.assertEqual(record["revision"], 1)
        self.assertEqual(record["corrections"][0]["text"], "old text")
        with self.assertRaises(RevisionConflict):
            correct_segment(self.state, 4, "stale", 0)

    def test_missing(self):
        with self.assertRaises(KeyError):
            correct_segment(self.state, 3, "text", 0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            correct_segment(self.state, 4, "   ", 0)

    def test_idempotent_edit(self):
        self.assertEqual(correct_segment(self.state, 4, "old text", 0).get("revision", 0), 0)

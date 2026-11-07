"""Offline safety checks for public reference corpus acquisition."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave
import zipfile

from tools.evaluation.corpora.prepare_meeting_reference_corpus import (
    MEETINGS, inspect_annotations, inspect_wav, sha256_file, verified_file,
)


class ReferenceCorpusTests(unittest.TestCase):
    def test_reference_meetings_are_distinct_sessions(self):
        self.assertEqual(len(MEETINGS), 3)
        self.assertEqual(len(set(mid[:2] for mid in MEETINGS)), 3)

    def test_existing_unexpected_bytes_fail_closed(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "audio.wav"
            path.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "refusing to replace"):
                verified_file(path, "a" * 64)
            self.assertEqual(sha256_file(path), sha256_file(path))
            self.assertEqual(path.read_bytes(), b"invalid")

    def test_wav_quality_preflight_rejects_tiny_clip(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "short.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"\0\0" * 16000)
            with self.assertRaisesRegex(ValueError, "Unexpected meeting audio"):
                inspect_wav(path)

    def test_annotation_inventory_requires_all_meetings(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "manual.zip"
            with zipfile.ZipFile(path, "w") as z:
                for mid in MEETINGS[:2]:
                    z.writestr("words/" + mid + ".A.words.xml", "<root />")
                    z.writestr("segments/" + mid + ".A.segments.xml", "<root />")
            with self.assertRaisesRegex(ValueError, "Missing original manual"):
                inspect_annotations(path)

    def test_annotation_inventory_counts_original_xml(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "manual.zip"
            with zipfile.ZipFile(path, "w") as z:
                for mid in MEETINGS:
                    z.writestr("ami_public_manual_1.6.2/words/" + mid + ".A.words.xml", "<root />")
                    z.writestr("ami_public_manual_1.6.2/segments/" + mid + ".A.segments.xml", "<root />")
            result = inspect_annotations(path)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(row["word_xml_files"] > 0 for row in result.values()))


if __name__ == "__main__":
    unittest.main()

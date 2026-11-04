"""AMI manual word-reference extraction must never invent audible words."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from tools.ami_human_text_reference import (
    START_S, END_S, MEETING, parse_ami_word_xml,
    words_from_official_archive, write_reference, SCHEMA,
)


def speaker_xml(words: list[tuple[str, str, str]]) -> bytes:
    rows = "".join(
        f'<w starttime="{start}" endtime="{end}">{text}</w>'
        for start, end, text in words
    )
    return ('<root xmlns:nite="http://example.org/nite">'
            + rows
            + '<vocalsound starttime="1510" endtime="1511">laugh</vocalsound>'
            + '</root>').encode("utf-8")


class AMIManualWordReferenceTests(unittest.TestCase):
    def test_exact_clip_and_lexical_words_only(self):
        xml = speaker_xml([
            ("1499.70", "1500.20", "truncated"),
            ("1500.10", "1500.50", "Hello"),
            ("1500.60", "1501.00", "there"),
            ("1740.00", "1740.20", "outside"),
        ])
        parsed = parse_ami_word_xml(xml, speaker="A")
        self.assertEqual([word.text for word in parsed], ["Hello", "there"])
        self.assertEqual(parsed[0].start_s, 1500.10)

    def test_real_archive_names_and_overlap_from_different_speakers(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "human.zip"
            with ZipFile(archive, "w", ZIP_DEFLATED) as writer:
                writer.writestr(
                    "ami_public_manual_1.6.2/words/TS3003b.A.words.xml",
                    speaker_xml([
                        ("1500.10", "1500.50", "Hello"),
                        ("1500.60", "1500.90", "there"),
                    ]),
                )
                writer.writestr(
                    "ami_public_manual_1.6.2/words/TS3003b.B.words.xml",
                    speaker_xml([
                        ("1500.30", "1500.70", "Yes"),
                        ("1501.50", "1501.90", "okay"),
                    ]),
                )
                writer.writestr(
                    "ami_public_manual_1.6.2/words/TS3003c.A.words.xml",
                    speaker_xml([("1500.20", "1500.40", "unrelated")]),
                )
            words, metadata = words_from_official_archive(archive)
            self.assertEqual(MEETING, "TS3003b")
            self.assertEqual(END_S - START_S, 240)
            self.assertEqual(
                [word.text for word in words],
                ["Hello", "Yes", "there", "okay"],
            )
            self.assertEqual(metadata["overlapping_word_entries"], 3)
            self.assertEqual(metadata["speaker_word_counts"], {"A": 2, "B": 2})
            self.assertNotIn("unrelated", json.dumps(metadata))
            reference = Path(folder) / "reference.txt"
            report = Path(folder) / "report.json"
            written = write_reference(
                words, metadata, reference=reference, report=report,
            )
            self.assertEqual(
                reference.read_text(encoding="utf-8"),
                "Hello Yes there okay\n",
            )
            self.assertEqual(written["schema"], SCHEMA)
            self.assertFalse(written["verified_against_mixed_audio"])
            self.assertNotIn("Hello Yes there okay", report.read_text())
            with self.assertRaises(FileExistsError):
                write_reference(
                    words, metadata, reference=reference, report=report,
                )

    def test_empty_or_unrelated_archive_fails_without_fabrication(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "wrong.zip"
            with ZipFile(archive, "w") as writer:
                writer.writestr("words/TS3003a.A.words.xml",
                                speaker_xml([("1500.1", "1500.5", "No")]))
            with self.assertRaisesRegex(ValueError, "lacks multiple"):
                words_from_official_archive(archive)

    def test_malformed_timestamp_and_speaker_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            parse_ami_word_xml(
                speaker_xml([("NaN", "1500.50", "unsafe")]),
                speaker="A",
            )
        with self.assertRaisesRegex(ValueError, "speaker ID"):
            parse_ami_word_xml(
                speaker_xml([("1500.10", "1500.50", "hello")]),
                speaker="../escape",
            )


if __name__ == "__main__":
    unittest.main()

"""Offline tests for selecting and preparing AMI shadow verification audio.

The fixture is artificial PCM used solely for protocol tests. The shipped
AMI case IDs and start offsets are grounded in the public RTTM files.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tools.ami_shadow10_prepare import (
    SELECTED, DURATION_S, crop_pcm16, parse_reference,
    prepare_case, validate_reference, write_inventory,
)


class AmiShadow10Tests(unittest.TestCase):
    def test_ten_pinned_four_minute_spans_are_distinct(self):
        self.assertEqual(len(SELECTED), 10)
        self.assertEqual(len({meeting for meeting, _ in SELECTED}), 10)
        self.assertEqual(
            {meeting[:2] for meeting, _ in SELECTED},
            {"EN", "ES", "IS", "TS"}
        )
        self.assertTrue(all(offset >= 0 for _, offset in SELECTED))

    def test_reference_crop_preserves_overlap_and_local_identity(self):
        raw = "\n".join([
            "SPEAKER EN2002b 1 298.5 4.0 <NA> <NA> PERSON_A <NA> <NA>",
            "SPEAKER EN2002b 1 300.25 2.0 <NA> <NA> PERSON_B <NA> <NA>",
            "SPEAKER EN2002b 1 539.0 3.0 <NA> <NA> PERSON_A <NA> <NA>",
            "SPEAKER EN2002b 1 541.0 2.0 <NA> <NA> PERSON_A <NA> <NA>",
        ])
        turns = parse_reference(raw, "EN2002b", 300)
        self.assertEqual(turns, [
            {"start_ms": 0, "end_ms": 2500, "speaker": "PERSON_A"},
            {"start_ms": 250, "end_ms": 2250, "speaker": "PERSON_B"},
            {"start_ms": 239000, "end_ms": 240000, "speaker": "PERSON_A"},
        ])
        with self.assertRaisesRegex(ValueError, "RTTM"):
            parse_reference(raw.replace("EN2002b", "EN2002a"), "EN2002b", 300)

    def test_validation_requires_two_speakers_with_repeat_turns(self):
        enough = [
            {"start_ms": index * 10000, "end_ms": index * 10000 + 4000,
             "speaker": who}
            for who in ("PERSON_A", "PERSON_B")
            for index in range(8)
        ]
        result = validate_reference(enough)
        self.assertEqual(result["reference_speaker_count"], 2)
        self.assertEqual(result["qualified_speaker_count"], 2)
        self.assertEqual(result["annotated_turn_count"], 16)
        with self.assertRaisesRegex(ValueError, "two frequently speaking"):
            validate_reference(enough[:8])

    def test_pcm_crop_uses_exact_frame_offset_and_duration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "original.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(
                    np.full(16000 * 250, 1123, dtype="<i2").tobytes()
                )
            target = root / "part.wav"
            crop_pcm16(source, target, 5)
            with wave.open(str(target), "rb") as reader:
                self.assertEqual(reader.getnframes(), 16000 * DURATION_S)
                self.assertEqual(reader.readframes(1), (1123).to_bytes(
                    2, "little", signed=True
                ))
            with self.assertRaisesRegex(ValueError, "ends before"):
                crop_pcm16(source, root / "too_late.wav", 15)

    def test_metadata_replay_and_manifest_keep_reference_turn_list(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meeting, offset = "EN2002b", 300
            full_audio = root / "source_fixture.wav"
            with wave.open(str(full_audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(
                    np.full(16000 * (offset + 241), 7000, dtype="<i2").tobytes()
                )
            annotation = root / "source_fixture.rttm"
            rows = []
            for who in ("PERSON_A", "PERSON_B"):
                for index in range(8):
                    start = offset + 20 * index
                    rows.append(
                        f"SPEAKER {meeting} 1 {start}.0 5.0 "
                        f"<NA> <NA> {who} <NA> <NA>"
                    )
            annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")

            def fake_download(_url, output):
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(full_audio, output)

            with patch(
                "tools.ami_shadow10_prepare.download_rttm",
                return_value=annotation,
            ):
                first = prepare_case(root, meeting, offset,
                                     downloader=fake_download)
                again = prepare_case(root, meeting, offset,
                                     downloader=fake_download)
            self.assertEqual(first, again)
            self.assertEqual(len(first["reference_turns"]), 16)
            self.assertEqual(first["annotated_turn_count"], 16)
            self.assertTrue((root / first["audio"]).exists())
            self.assertFalse((root / "cache" / f"{meeting}.Mix-Headset.wav").exists())
            inventory = write_inventory(root)
            self.assertEqual(inventory["prepared_count"], 1)
            self.assertFalse(inventory["complete"])
            self.assertEqual(inventory["cases"][0]["reference_turns"],
                             first["reference_turns"])
            # Cached data is integrity checked, not overwritten.
            (root / first["audio"]).write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_case(root, meeting, offset, downloader=fake_download)


if __name__ == "__main__":
    unittest.main()

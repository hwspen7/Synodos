"""Post-meeting second-opinion diarization is bounded and does not mutate text."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from speech.offline_speaker import (
    OfflineDiarizationUnavailable, normalize_turns, run_offline_refinement,
    validate_recording,
)
from speech.text_format import split_display_sentences, ends_finished_thought
from storage.meetings import MeetingStore


class FakeDiarizer:
    def __call__(self, filename: str):
        assert filename.endswith(".wav")
        return SimpleNamespace(exclusive_speaker_diarization=[
            (SimpleNamespace(start=0.0, end=2.0), "SPEAKER_00"),
            (SimpleNamespace(start=2.1, end=3.2), "SPEAKER_01"),
            (SimpleNamespace(start=3.3, end=4.0), "SPEAKER_00"),
        ])


class OfflineSpeakerTests(unittest.TestCase):
    def test_second_model_produces_alternating_turns_without_modifying_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mid = "a" * 32
            filename = "b" * 32 + ".wav"
            folder = root / mid
            folder.mkdir()
            (folder / filename).write_bytes(b"RIFF" + b"0" * 100)
            data = run_offline_refinement(root, mid, filename, backend=FakeDiarizer())
            self.assertTrue(data["review_required"])
            self.assertFalse(data["auto_applied"])
            self.assertTrue(data["relative_to_recording"])
            self.assertEqual(data["speaker_count"], 2)
            self.assertEqual(
                [x["speaker"] for x in data["speaker_turns"]],
                ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
            )
            self.assertEqual(data["speaker_turns"][1]["start_ms"], 2100)

    def test_rejects_path_escape_and_invalid_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            for filename in ("../secret.wav", "some.wav", "a" * 32 + ".mp3"):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    validate_recording(Path(temp), "a" * 32, filename)
            with self.assertRaises(ValueError):
                validate_recording(Path(temp), "../private", "a" * 32 + ".wav")

    def test_no_token_or_uninstalled_model_reports_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mid, filename = "a" * 32, "b" * 32 + ".wav"
            (root / mid).mkdir()
            (root / mid / filename).write_bytes(b"RIFF")
            with patch.dict("os.environ", {"HF_TOKEN": "", "HUGGINGFACE_TOKEN": ""}):
                with self.assertRaisesRegex(OfflineDiarizationUnavailable, "HF_TOKEN"):
                    run_offline_refinement(root, mid, filename)

    def test_malformed_spans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            normalize_turns([("not-a-segment", "SPEAKER_00")])
        self.assertEqual(normalize_turns([
            (SimpleNamespace(start=1.0, end=0.5), "SPEAKER_00"),
            (SimpleNamespace(start=0.0, end=1.0), "SPEAKER_01"),
        ]), [{"start_ms": 0, "end_ms": 1000, "speaker": "SPEAKER_01"}])


class ExportParagraphParityTests(unittest.TestCase):
    def test_export_does_not_merge_two_finished_sentences_or_unknown_speakers(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MeetingStore(Path(temp))
            mid = store.create(title="Test")["id"]
            values = [
                ("Really, what a coincidence.", 0, 10000, 1),
                ("We laughed a lot.", 10300, 19000, 1),
                ("Absolutely.", 19400, 25000, 1),
                ("Hey, how are you?", 26000, 30000, -1),
                ("Hello, how are you?", 30200, 37000, -1),
            ]
            state = {
                "id": mid, "title": "Test", "sample_rate": 16000,
                "merge_gap_s": 2.0, "segments": [{
                    "id": i + 1, "text": text, "start_sample": a, "end_sample": b,
                    "speaker_id": sid, "speaker_name": "Speaker 1" if sid == 1 else "Unidentified Speaker",
                    "cluster": sid, "emb": None,
                } for i, (text, a, b, sid) in enumerate(values)],
            }
            store.save_state(state)
            import json
            paragraphs = json.loads(store.export(mid, "json"))["transcript"]["paragraphs"]
            self.assertEqual(len(paragraphs), 5)
            self.assertEqual(paragraphs[0]["sentences"], ["Really, what a coincidence."])
            self.assertIn("We laughed a lot.", store.export(mid, "md"))

    def test_legacy_multi_sentence_export_preserves_every_sentence(self):
        text = "Really, what a coincidence. We laughed a lot. Absolutely. Hey, how are you?"
        self.assertEqual(len(split_display_sentences(text)), 4)
        self.assertTrue(ends_finished_thought(text))
        self.assertEqual(" ".join(split_display_sentences(text)), text)


if __name__ == "__main__":
    unittest.main()

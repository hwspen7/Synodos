"""Offline tests for independent AMI word overlap and private review queue."""
from __future__ import annotations
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from tools.evaluation.corpora.prepare_audio_review_queue import (
    annotation_stats, classify_window, extract_wav_clip, make_review_rows,
    select_windows, word_overlap_indices,
)


class AudioReviewQueueTests(unittest.TestCase):
    def test_word_overlap_is_speaker_specific(self):
        words = [
            {"speaker": "A", "start_ms": 0, "end_ms": 600, "text": "one"},
            {"speaker": "B", "start_ms": 300, "end_ms": 700, "text": "two"},
            {"speaker": "A", "start_ms": 900, "end_ms": 1200, "text": "three"},
        ]
        flagged = word_overlap_indices(words)
        self.assertEqual(flagged, {0, 1})
        stats = annotation_stats(words, flagged)
        self.assertEqual(stats["other_speaker_overlap_tokens"], 2)
        self.assertEqual(stats["nonoverlap_nonzero_reference_tokens"], 1)

    def test_same_speaker_is_not_overlap(self):
        words = [
            {"speaker": "A", "start_ms": 0, "end_ms": 600, "text": "one"},
            {"speaker": "A", "start_ms": 300, "end_ms": 700, "text": "two"},
        ]
        self.assertFalse(word_overlap_indices(words))

    def test_zero_duration_not_fabricated(self):
        words = [
            {"speaker": "A", "start_ms": 0, "end_ms": 0, "text": "zero"},
            {"speaker": "B", "start_ms": 0, "end_ms": 100, "text": "spoken"},
        ]
        self.assertEqual(annotation_stats(words, word_overlap_indices(words)),
                         {
                             "annotated_reference_tokens": 2,
                             "other_speaker_overlap_tokens": 0,
                             "zero_duration_reference_tokens": 1,
                             "nonoverlap_nonzero_reference_tokens": 1,
                             "other_speaker_overlap_fraction": 0,
                             "nonoverlap_nonzero_fraction": .5,
                         })

    def test_review_classification_keeps_uncertain_independent(self):
        row = dict(reference_words=9, predicted_words=8, screening_wer=.5)
        self.assertEqual(classify_window(row, 0), "CLEAN_DISAGREEMENT")
        self.assertEqual(classify_window(row, .2), "CROSS_TALK")
        row["predicted_words"] = 0
        self.assertEqual(classify_window(row, 0), "POSSIBLE_DELETION")
        row["reference_words"] = 0
        row["predicted_words"] = 8
        self.assertEqual(classify_window(row, 0), "POSSIBLE_INSERTION")

    def test_window_overlap_annotation_reuses_real_words(self):
        reference = [
            {"speaker": "A", "start_ms": 0, "end_ms": 600, "text": "alpha"},
            {"speaker": "B", "start_ms": 300, "end_ms": 700, "text": "beta"},
        ]
        predictions = [
            {"id": 0, "text": "alpha beta", "start_ms": 0, "end_ms": 1200}
        ]
        rows, stats = make_review_rows("EN2002a", 10, reference, predictions, 10)
        self.assertEqual(rows[0]["category"], "CROSS_TALK")
        self.assertEqual(rows[0]["reference_overlap_fraction"], 1)
        self.assertEqual(stats["other_speaker_overlap_tokens"], 2)

    def test_balanced_selection_has_unique_windows(self):
        rows = []
        for i in range(36):
            name = ("CLEAN_AGREEMENT", "CLEAN_DISAGREEMENT", "CROSS_TALK")[i % 3]
            rows.append(dict(
                meeting_id="EN2002a", category=name, start_s=i * 10,
                reference_words=10, predicted_words=10,
                screening_wer=.1 if name == "CLEAN_AGREEMENT" else .6,
                reference_overlap_fraction=1 if name == "CROSS_TALK" else 0,
            ))
        selected = select_windows(rows, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({x["start_s"] for x in selected}), 12)
        self.assertTrue({"CLEAN_AGREEMENT", "CLEAN_DISAGREEMENT", "CROSS_TALK"}
                        .issubset({x["category"] for x in selected}))

    def test_clip_is_raw_pcm_and_duration_is_exact(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "source.wav"
            out = Path(temp) / "clip.wav"
            samples = b"\x01\x00" * (16000 * 4)
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples)
            self.assertEqual(extract_wav_clip(path, out, 1, 3), 2)
            with wave.open(str(out), "rb") as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), samples[32000:96000])

    def test_queue_does_not_approve_human_gold(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / "review_queue.csv"
            with file.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio_review_status"])
                writer.writeheader()
                writer.writerow({"audio_review_status": "UNREVIEWED"})
            self.assertIn("UNREVIEWED", file.read_text())


if __name__ == "__main__":
    unittest.main()

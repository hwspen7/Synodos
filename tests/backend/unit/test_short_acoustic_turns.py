"""Regression contracts for acoustic splits inside one Whisper utterance.

These tests are synthetic: two distinct voices are represented by known
embedding prototypes. They validate losslessness and short-turn detection,
not real-world speaker diarization error rates.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor
from speech.config import SpeechSettings
from speech.text_format import join_transcript_text
from speech.turn_split import split_by_acoustic_turns


RATE = 16000


def voice_audio(voice_ids: tuple[int, ...], seconds_per_voice: float) -> np.ndarray:
    duration = round(seconds_per_voice * RATE)
    return np.concatenate([
        np.full(duration, 0.2 if voice == 1 else -0.2, dtype=np.float32)
        for voice in voice_ids
    ])


class SyntheticSpeakerEmbedding:
    def __init__(self):
        self.calls = 0

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        if float(np.mean(samples)) > 0.02:
            return np.array([1.0, 0.0], dtype=np.float32)
        if float(np.mean(samples)) < -0.02:
            return np.array([0.0, 1.0], dtype=np.float32)
        # Force a conservative identity abstention for mixed windows.
        return np.array([0.0, 0.0], dtype=np.float32)


class ShortMixedUtteranceTests(unittest.TestCase):
    def test_new_speaker_requires_two_independent_acoustic_turns(self):
        from sessions.speaker_birth_guard import tentative_new_clusters
        def row(number, a, b, quality="window_estimate"):
            return {
                "id": number, "cluster": 2, "speaker_id": -number,
                "emb": np.array([1.0, 0.0], dtype=np.float32),
                "speaker_uncertain": False, "start_sample": a,
                "end_sample": b, "timestamp_quality": quality, "words": [],
            }
        first = row(1, 0, 16000)
        self.assertEqual(tentative_new_clusters([first], set()), {2})
        # A duplicated prediction of the same time range is not new evidence.
        overlap = row(2, 0, 16000)
        self.assertEqual(
            tentative_new_clusters([first, overlap], set()), {2}
        )
        second = row(3, 32000, 48000)
        self.assertEqual(
            tentative_new_clusters([first, second], set()), set()
        )
        first["speaker_id"] = 5
        self.assertEqual(tentative_new_clusters([first], {5}), set())

    def test_contiguous_tail_fragments_cannot_birth_a_new_speaker(self):
        from sessions.speaker_birth_guard import tentative_new_clusters
        rows = [
            {"id": 1, "cluster": 4, "speaker_id": -1,
             "emb": np.array([1.0, 0.0], dtype=np.float32),
             "speaker_uncertain": False,
             "start_sample": 0, "end_sample": RATE * 2,
             "timestamp_quality": "window_estimate", "words": []},
            {"id": 2, "cluster": 4, "speaker_id": -2,
             "emb": np.array([1.0, 0.0], dtype=np.float32),
             "speaker_uncertain": False,
             "start_sample": RATE * 2 + 1200,
             "end_sample": RATE * 3,
             "timestamp_quality": "window_estimate", "words": []},
        ]
        self.assertEqual(tentative_new_clusters(rows, set()), {4})
        rows.append(dict(rows[1], id=3, start_sample=RATE * 5,
                         end_sample=RATE * 6))
        self.assertEqual(tentative_new_clusters(rows, set()), set())

    def test_new_speaker_single_long_precise_turn_can_be_verified(self):
        from sessions.speaker_birth_guard import tentative_new_clusters
        case = {
            "id": 1, "cluster": 0, "speaker_id": -1,
            "emb": np.array([1.0, 0.0], dtype=np.float32),
            "speaker_uncertain": False,
            "start_sample": 0, "end_sample": 64000,
            "timestamp_quality": "word",
            "words": [{"text": "this"}, {"text": "is"}, {"text": "speech"}],
        }
        self.assertEqual(tentative_new_clusters([case], set()), set())
        case["timestamp_quality"] = "window_estimate"
        self.assertEqual(tentative_new_clusters([case], set()), {0})


    def test_live_two_speakers_do_not_promote_unsupported_third(self):
        class ClusterModels:
            def cluster(self, embeddings):
                self_size = len(embeddings)
                self_test_labels = [0, 1, 0, 1, 2]
                assert self_size == 5
                return self_test_labels

        def segment(number, speaker, start):
            return {
                "id": number, "cluster": -3, "speaker_id": -number,
                "speaker_name": "Unidentified Speaker",
                "text": "recognized utterance", "language": "en",
                "emb": np.array(
                    [1.0, 0.0] if speaker == "A"
                    else [0.0, 1.0] if speaker == "B"
                    else [0.7, 0.7], dtype=np.float32
                ),
                "speaker_uncertain": False,
                "speaker_evidence_status": "turn_voiceprint",
                "start_sample": start,
                "end_sample": start + RATE * 2,
                "timestamp_quality": "window_estimate",
                "words": [],
            }

        with patch("sessions.processor.load_settings",
                   return_value=SpeechSettings(
                       engine="hybrid",
                       hybrid_qwen_python="E:/ASR/python.exe"
                   )):
            processor = SessionProcessor(
                "two-voices-plus-one-candidate", AudioBuffer(RATE),
                Settings(), ClusterModels(), asyncio.Queue(),
            )
        processor.segments = [
            segment(1, "A", 0),
            segment(2, "B", RATE * 3),
            segment(3, "A", RATE * 6),
            segment(4, "B", RATE * 9),
            segment(5, "C", RATE * 12),
        ]
        processor._recluster()
        self.assertEqual(
            len({row["speaker_id"] for row in processor.segments
                 if row["speaker_id"] > 0}), 2
        )
        self.assertEqual(
            processor.segments[4]["speaker_name"],
            "Unidentified Speaker"
        )
        self.assertEqual(
            [row["text"] for row in processor.segments],
            ["recognized utterance"] * 5
        )

    @staticmethod
    def sentence(with_words: bool = True) -> dict:
        words = [
            {"start_ms": 100, "end_ms": 300, "text": "Hello"},
            {"start_ms": 350, "end_ms": 800, "text": " there."},
            {"start_ms": 1320, "end_ms": 1600, "text": " Hi"},
            {"start_ms": 1700, "end_ms": 2260, "text": " again."},
        ]
        return {
            "start": 0, "end": 2400,
            "text": "Hello there. Hi again.",
            "words": words if with_words else [],
        }

    def test_two_one_second_voices_within_one_asr_segment_are_split(self):
        audio = voice_audio((1, 2), 1.2)
        embed = SyntheticSpeakerEmbedding()
        result = split_by_acoustic_turns(self.sentence(), audio, embed)
        self.assertEqual(embed.calls, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual([part["text"] for part in result], [
            "Hello there.", "Hi again.",
        ])
        self.assertEqual([(part["start"], part["end"]) for part in result], [
            (100, 800), (1320, 2260),
        ])
        self.assertTrue(all(part["ambiguous"] for part in result))
        self.assertTrue(all(part["emb"] is None for part in result))
        self.assertEqual(join_transcript_text(
            result[0]["text"], result[1]["text"]),
            self.sentence()["text"],
        )

    def test_single_voice_variation_does_not_create_false_turn(self):
        audio = voice_audio((1, 1), 1.2)
        embed = SyntheticSpeakerEmbedding()
        result = split_by_acoustic_turns(self.sentence(), audio, embed)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], self.sentence()["text"])
        self.assertFalse(result[0]["ambiguous"])
        self.assertIsNotNone(result[0]["emb"])

    def test_no_word_timestamps_abstains_without_inventing_turn_times(self):
        audio = voice_audio((1, 2), 1.2)
        result = split_by_acoustic_turns(
            self.sentence(with_words=False), audio, SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[0]["end"], 2400)
        self.assertEqual(result[0]["text"], self.sentence()["text"])
        self.assertTrue(result[0]["ambiguous"])

    def test_inconsistent_alignment_does_not_drop_original_words(self):
        source = self.sentence()
        source["words"][0]["text"] = "Entirely unrelated"
        parts = split_by_acoustic_turns(
            source, voice_audio((1, 2), 1.2), SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["text"], source["text"])
        self.assertTrue(parts[0]["ambiguous"])

    def test_word_alignment_cannot_alter_confirmed_punctuation(self):
        source = self.sentence()
        source["words"][1]["text"] = " there"
        pieces = split_by_acoustic_turns(
            source, voice_audio((1, 2), 1.2), SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["text"], "Hello there. Hi again.")
        self.assertTrue(pieces[0]["ambiguous"])

    def test_near_matching_alignment_cannot_omit_one_spoken_word(self):
        source = self.sentence()
        source["text"] = "Hello there. Hi everybody again."
        pieces = split_by_acoustic_turns(
            source, voice_audio((1, 2), 1.2), SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["text"], source["text"])
        self.assertTrue(pieces[0]["ambiguous"])

    def test_longer_two_voice_turns_keep_distinct_consistent_embeddings(self):
        sentence = {
            "start": 0, "end": 4000,
            "text": "One participant speaking. A second participant replying.",
            "words": [
                {"start_ms": 100, "end_ms": 900,
                 "text": "One participant"},
                {"start_ms": 1000, "end_ms": 1800,
                 "text": " speaking."},
                {"start_ms": 2200, "end_ms": 3000,
                 "text": " A second participant"},
                {"start_ms": 3100, "end_ms": 3900,
                 "text": " replying."},
            ],
        }
        result = split_by_acoustic_turns(
            sentence, voice_audio((1, 1, 2, 2), 1.0),
            SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "One participant speaking.")
        self.assertEqual(result[1]["text"], "A second participant replying.")
        self.assertFalse(result[0]["ambiguous"])
        self.assertFalse(result[1]["ambiguous"])
        self.assertGreater(float(result[0]["emb"][0]), 0.99)
        self.assertGreater(float(result[1]["emb"][1]), 0.99)
        self.assertEqual(join_transcript_text(
            result[0]["text"], result[1]["text"]),
            sentence["text"],
        )

    def test_async_acoustic_split_preserves_all_confirmed_text_and_timestamps(self):
        audio = voice_audio((1, 2), 1.2)
        source = self.sentence()
        pieces = split_by_acoustic_turns(
            source, audio, SyntheticSpeakerEmbedding(),
        )
        self.assertEqual(len(pieces), 2)
        cfg = Settings()

        class Models:
            def embed(self, _audio):
                return None

        with patch("sessions.processor.load_settings", return_value=SpeechSettings(
                engine="whisper", model="large-v3")):
            processor = SessionProcessor(
                "short-turn-contract", AudioBuffer(RATE), cfg,
                Models(), asyncio.Queue(),
            )
        events = []
        processor._put = events.append
        processor.segments = [{
            "id": 1,
            "_audio_base_sample": 0,
            "start_sample": 0,
            "end_sample": int(2.4 * RATE),
            "text": source["text"],
            "language": "en",
            "words": source["words"],
            "emb": None,
            "cluster": -3,
            "speaker_uncertain": True,
        }]
        processor._next_segment_id = 2
        done = Future()
        done.set_result(pieces)
        processor._speaker_tasks[1] = done
        self.assertTrue(processor._collect_speakers())
        self.assertEqual(processor._acoustic_turn_splits, 1)
        self.assertEqual(len(processor.segments), 2)
        self.assertEqual(
            [s["id"] for s in processor.segments], [1, 2],
        )
        self.assertEqual([
            (round(s["start_sample"] * 1000 / RATE),
             round(s["end_sample"] * 1000 / RATE))
            for s in processor.segments
        ], [(100, 800), (1320, 2260)])
        self.assertEqual([
            len(s["words"]) for s in processor.segments
        ], [2, 2])
        processor._recluster()
        self.assertTrue(processor._rebuild_and_push())
        snapshots = [event for event in events
                     if event.get("type") == "transcript"]
        self.assertEqual(len(snapshots), 1)
        out = snapshots[0]["segments"]
        self.assertEqual(len(out), 2)
        self.assertEqual(join_transcript_text(
            out[0]["text"], out[1]["text"]
        ), source["text"])
        self.assertEqual(
            [(s["start_ms"], s["end_ms"]) for s in out],
            [(100, 800), (1320, 2260)],
        )
        self.assertTrue(all(
            s["speaker_name"] == "Unidentified Speaker" for s in out
        ))

    def test_invalid_window_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, "sampling bounds"):
            split_by_acoustic_turns(
                self.sentence(), voice_audio((1, 2), 1.2),
                SyntheticSpeakerEmbedding(), min_window_s=0.1,
            )
        with self.assertRaisesRegex(ValueError, "sampling bounds"):
            split_by_acoustic_turns(
                self.sentence(), voice_audio((1, 2), 1.2),
                SyntheticSpeakerEmbedding(), max_windows=100,
            )


if __name__ == "__main__":
    unittest.main()

"""Shadow ASR selection tests without Qwen weights, Whisper models or GPU."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import wave

from tools.evaluation.asr.qwen_shadow_review import select_gaps, shadow


class FakeTranscriber:
    def __init__(self):
        self.calls = 0
    def transcribe(self, *, audio, language):
        self.calls += 1
        assert len(audio[0]) <= 160000
        assert language == "English"
        return [SimpleNamespace(text="Test candidate", language="English")]


class ShadowASRTests(unittest.TestCase):
    def test_vad_supported_gap_is_not_a_reference_or_auto_commit(self):
        final = [{"start_ms": 0, "end_ms": 500, "text": "one"}]
        vad = [{"start": 0, "end": 8000},
               {"start": 16000, "end": 32000}]
        result = select_gaps(vad, final, 2000, max_windows=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["reason"],
                         "vad_speech_without_confirmed_caption")

    def test_full_shadow_runs_with_injected_decoders(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "clip.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 32000)
            audit = folder / "segment_audit.json"
            audit.write_text(json.dumps({
                "schema": "meeting-live-caption-segment-audit-v1",
                "source_audio_duration_s": 2.0,
                "source_audio": "clip.wav",
                "final_segments": [
                    {"start_ms": 0, "end_ms": 400, "text": "known"},
                ],
            }), encoding="utf-8")
            worker = FakeTranscriber()
            output = shadow(
                audio, audit, language="en", infer=True,
                vad_provider=lambda _: [{"start": 16000, "end": 32000}],
                transcriber=worker,
            )
            self.assertEqual(worker.calls, 1)
            self.assertFalse(output["auto_replaced_live_text"])
            self.assertFalse(output["reference_words_used_to_select_windows"])
            self.assertEqual(output["candidate_transcripts"][0]["candidate_text"],
                             "Test candidate")
            self.assertTrue(output["candidate_transcripts"][0]["human_review_required"])

    def test_incorrect_audio_or_malformed_segments_fail(self):
        with self.assertRaises(ValueError):
            select_gaps([], [{"start_ms": 500, "end_ms": 100,
                              "text": "bad"}], 2000)
        with self.assertRaises(ValueError):
            select_gaps([], [], 2000, max_windows=40)


if __name__ == "__main__":
    unittest.main()

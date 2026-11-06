"""Offline safety and provenance tests for real meeting audio evaluation."""
from __future__ import annotations

import json
import os
import sys
import types
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave
import zipfile

from tools.evaluation.corpora.prepare_real_meeting_evaluation import (
    PREDICTION_SCHEMA, REFERENCE_SCHEMA, check_root, evaluation_id,
    make_state, parse_reference_archive, transcribe, validate_prediction, write_private_json,
)


class RealMeetingEvaluationTests(unittest.TestCase):
    def sample_prediction(self):
        return {"schema": PREDICTION_SCHEMA, "meeting": "ES2004a",
                "source_audio_sha256": "b" * 64, "model": "large-v3",
                "pipeline": "speech.asr.whisper.WhisperEngine",
                "reference_used_for_inference": False,
                "independently_audio_reviewed": False,
                "segments": [{"id": i, "start_ms": i * 1000,
                              "end_ms": i * 1000 + 600,
                              "text": f"spoken audio number {i}", "language": "en"}
                             for i in range(11)]}

    def test_independent_meeting_ids(self):
        self.assertEqual(evaluation_id("EN2002a"), "ami_eval_en2002a")
        self.assertNotEqual(evaluation_id("IS1009a"), evaluation_id("ES2004a"))
        with self.assertRaises(ValueError):
            evaluation_id("TS3003b")

    def test_real_prediction_contract(self):
        doc = self.sample_prediction()
        rows = validate_prediction(doc, "ES2004a", "b" * 64, 20)
        self.assertEqual(len(rows), 11)

    def test_never_accepts_official_reference_as_model_prediction(self):
        doc = self.sample_prediction()
        doc["reference_used_for_inference"] = True
        with self.assertRaises(ValueError):
            validate_prediction(doc, "ES2004a", "b" * 64, 20)
        doc = self.sample_prediction()
        doc["independently_audio_reviewed"] = True
        with self.assertRaises(ValueError):
            validate_prediction(doc, "ES2004a", "b" * 64, 20)

    def test_rejects_invalid_and_disordered_timestamps(self):
        doc = self.sample_prediction()
        doc["segments"][3]["start_ms"] = 100
        with self.assertRaises(ValueError):
            validate_prediction(doc, "ES2004a", "b" * 64, 20)
        doc = self.sample_prediction()
        doc["segments"][0]["end_ms"] = 99_999
        with self.assertRaises(ValueError):
            validate_prediction(doc, "ES2004a", "b" * 64, 20)

    def test_unverified_speaker_not_fabricated(self):
        doc = self.sample_prediction()
        meta, state = make_state("ES2004a", 20, doc["segments"], doc)
        self.assertEqual(meta["speakers"], 0)
        self.assertTrue(all(x["speaker_id"] == -1 for x in state["segments"]))
        self.assertFalse(state["human_audio_reviewed"])
        self.assertEqual(state["segments"][2]["start_sample"], 32000)

    def test_private_json_refuses_to_overwrite(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "private.json"
            write_private_json(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_private_json(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text())["x"], 1)

    def test_xml_speaker_words_are_distinct_from_gold(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "manual.zip"
            with zipfile.ZipFile(path, "w") as z:
                for speaker in ("A", "B"):
                    body = "<root>" + "".join(
                        f'<w starttime="{i / 10}" endtime="{i / 10 + .05}">alpha</w>'
                        for i in range(60)) + "</root>"
                    z.writestr(f"manual/words/ES2004a.{speaker}.words.xml", body)
            words, metadata = parse_reference_archive(path, "ES2004a", 20)
            self.assertEqual(len(words), 120)
            self.assertEqual(len(metadata["xml_sha256"]), 2)
            self.assertEqual({w["speaker"] for w in words}, {"A", "B"})


    def test_real_asr_reads_audio_windows_and_restores_absolute_time(self):
        # Use a fake recognizer; the actual GPU/acoustic model is not in CI.
        class FakeEngine:
            def __init__(self, *args, **kwargs):
                pass

            def recognize(self, samples, language="auto"):
                self_lang = language
                if self_lang != "en":
                    raise AssertionError("Unexpected recognition language")
                rows = tuple(types.SimpleNamespace(text=f"spoken word {i}",
                                       start_ms=500 + i * 4500,
                                       end_ms=1200 + i * 4500,
                                       language="en", timestamp_quality="segment",
                                       recognition_origin="primary") for i in range(7))
                return types.SimpleNamespace(utterances=rows, language="en", engine="whisper")

        with TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "test.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * (16000 * 60))
            fake_whisper = types.ModuleType("speech.asr.whisper")
            fake_whisper.WhisperEngine = FakeEngine
            import tools.evaluation.corpora.prepare_real_meeting_evaluation as module
            with (mock.patch.object(module, "source_audio", return_value=(audio, 60.0)),
                  mock.patch.object(module, "sha256_file", return_value="a" * 64),
                  mock.patch.dict(sys.modules, {"speech.asr.whisper": fake_whisper}),
                  mock.patch.dict(os.environ, {"HF_HOME": str(root)})):
                transcribe(root, "large-v3", "cpu", "int8", 30)
            prediction = json.loads((root / "predictions" / "EN2002a_whisper.json").read_text())
            segments = prediction["segments"]
            self.assertEqual(len(segments), 13)
            self.assertTrue(any(x["start_ms"] >= 30_000 for x in segments))
            self.assertEqual([x["id"] for x in segments], list(range(len(segments))))
            self.assertFalse(prediction["reference_used_for_inference"])


    def test_import_calls_psycopg3_cursor_executemany(self):
        # Regresses the real-world Connection.executemany AttributeError.
        # A fake Connection intentionally has no executemany method.
        import contextlib
        import tools.evaluation.corpora.prepare_real_meeting_evaluation as module
        from tools.evaluation.corpora.prepare_meeting_reference_corpus import MEETINGS

        class Cursor:
            def __init__(self):
                self.batches = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def executemany(self, sql, params):
                self.batches.append((sql, list(params)))

        class Connection:
            def __init__(self):
                self.info = types.SimpleNamespace(dbname="meeting_test")
                self.writes = []
                self.cur = Cursor()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                self.writes.append((sql, params))
                if sql.startswith("SELECT id FROM meeting_records"):
                    return types.SimpleNamespace(fetchall=lambda: [])
                return types.SimpleNamespace()

            def transaction(self):
                return contextlib.nullcontext()

            def cursor(self):
                return self.cur

        conn = Connection()
        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.connect = lambda *args, **kwargs: conn
        fake_types = types.ModuleType("psycopg.types")
        fake_json = types.ModuleType("psycopg.types.json")
        fake_json.Jsonb = lambda value: value
        doc = self.sample_prediction()
        reports = [(mid, Path("irrelevant.wav"), 20.0,
                    doc["segments"], doc) for mid in MEETINGS]
        with (
            mock.patch.dict(sys.modules, {
                "psycopg": fake_psycopg,
                "psycopg.types": fake_types,
                "psycopg.types.json": fake_json,
            }),
            mock.patch.object(module, "load_predictions", return_value=reports),
            mock.patch.object(module, "inspect_db_name"),
            mock.patch.dict(os.environ, {
                "DATABASE_URL": "postgresql://unused/meeting_test",
            }),
        ):
            module.import_predictions(Path("unused"), apply=True)

        self.assertEqual(len(conn.cur.batches), 3)
        self.assertEqual(sum(len(batch) for _, batch in conn.cur.batches), 33)
        self.assertEqual(
            [batch[0][0] for _, batch in conn.cur.batches],
            ["ami_eval_en2002a", "ami_eval_es2004a", "ami_eval_is1009a"],
        )
        self.assertTrue(all(
            "INSERT INTO meeting_segments" in sql
            for sql, batch in conn.cur.batches
        ))
        self.assertEqual(sum(
            sql.startswith("INSERT INTO meeting_records")
            for sql, _ in conn.writes
        ), 3)


if __name__ == "__main__":
    unittest.main()

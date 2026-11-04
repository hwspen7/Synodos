"""Contract tests for the first-stage isolated speaker model adapters.

No downloads, external models, GPU access, or private audio are required.
The real-model benchmark remains a separate operator-run acceptance gate.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from speech.speaker_backends import (
    CamPlusEmbedding, EcapaEmbedding, Nemotron3Diarization,
    create_diarization_backend, create_embedding_backend,
    normalize_nemotron_turns, unit_embedding, validated_audio,
)
from tools.speaker_stage1_compare import (
    MANIFEST_SCHEMA, baseline_turns, compare_diarization,
    compare_verification, complete, load_manifest, run,
    verification_metrics, verified_path,
)


def write_wave(root: Path, filename: str, level: float) -> dict:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        data = np.full(32000, round(level * 32767), dtype="<i2")
        wav.writeframes(data.tobytes())
    digest = sha256(path.read_bytes()).hexdigest()
    return {"audio": filename, "sha256": digest}


def prepare_diarization(root: Path) -> dict:
    audio = write_wave(root, "audio/clip.wav", 0.1)
    baseline = {
        "source_audio": r"benchmark\fixed195\audio\clip.wav",
        "source_audio_duration_s": 2.0,
        "delivery_matched": True,
        "server_closed": True,
        "errors": [],
        "flags": [],
        "final_cluster_diagnostics": {
            "anonymous_timeline": {
                "total_segments": 2,
                "exported_segments": 2,
                "truncated": False,
                "segments": [
                    {
                        "segment_id": 11, "start_ms": 0,
                        "end_ms": 1000, "published_speaker_id": 1,
                    },
                    {
                        "segment_id": 3, "start_ms": 1000,
                        "end_ms": 2000, "published_speaker_id": 2,
                    },
                ],
            },
        },
    }
    dest = root / "baseline.json"
    dest.write_text(json.dumps(baseline), encoding="utf-8")
    return {
        "id": "two-speaker-recording",
        **audio,
        "reference_turns": [
            {"start_ms": 0, "end_ms": 1000, "speaker": "alice"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "bob"},
        ],
        "camplus_replay": {
            "path": "baseline.json",
            "sha256": sha256(dest.read_bytes()).hexdigest(),
        },
    }


class DummyCampManager:
    def embed(self, audio):
        return np.array([3.0] + [4.0] + [0.0] * 190, dtype=np.float32)


class DummyDiarizer:
    name = "nemotron3"
    model_id = "fake/deterministic"

    def diarize(self, audio):
        assert len(audio) == 32000
        return [
            {"start_ms": 0, "end_ms": 1000, "speaker_id": 8},
            {"start_ms": 1000, "end_ms": 2000, "speaker_id": 4},
        ]


class FailedDiarizer:
    def diarize(self, audio):
        raise RuntimeError("Injected model unavailable")


class DummyEmbedder:
    model_id = "test/deterministic-embedding"

    def embed(self, audio):
        average = float(np.mean(audio))
        if average > 0:
            return unit_embedding(np.array([1.0, 0.0] + [0] * 30))
        return unit_embedding(np.array([0.0, 1.0] + [0] * 30))


class SpeakerStageOneTests(unittest.TestCase):
    def test_default_factory_does_not_load_any_optional_models(self):
        self.assertIsInstance(
            create_embedding_backend("camplus", manager=DummyCampManager()),
            CamPlusEmbedding,
        )
        self.assertIsInstance(create_embedding_backend("ecapa"), EcapaEmbedding)
        self.assertIsInstance(
            create_diarization_backend("nemotron3"), Nemotron3Diarization
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            create_embedding_backend("camplus")
        with self.assertRaises(ValueError):
            create_diarization_backend("unrecognized")

    def test_camplusplus_unit_normalizes_and_rejects_bad_input(self):
        adapter = create_embedding_backend(
            "camplus", manager=DummyCampManager()
        )
        emb = adapter.embed(np.ones(16000, dtype=np.float32) * 0.1)
        self.assertEqual(emb.shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0)
        self.assertAlmostEqual(float(emb[0]), 0.6)
        with self.assertRaisesRegex(ValueError, "mono"):
            validated_audio(np.ones((2, 16000), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validated_audio(np.array([float("nan")]))
        with self.assertRaisesRegex(ValueError, "normalized"):
            validated_audio(np.array([1500], dtype=np.float32))
        self.assertIsNone(unit_embedding(np.zeros(192, dtype=np.float32)))
        self.assertIsNone(
            unit_embedding(np.full(192, np.nan, dtype=np.float32))
        )

    def test_nemotron_turn_conversion_preserves_overlap_and_slots(self):
        predicted = normalize_nemotron_turns([
            {"Speaker": 1, "Start": 1.0, "End": 2.0},
            {"Speaker": 0, "Start": 0.0, "End": 1.5},
        ], duration_ms=2000)
        self.assertEqual(
            predicted, [
                {"start_ms": 0, "end_ms": 1500, "speaker_id": 1},
                {"start_ms": 1000, "end_ms": 2000, "speaker_id": 2},
            ],
        )
        with self.assertRaisesRegex(ValueError, "Invalid Nemotron"):
            normalize_nemotron_turns(
                [{"Speaker": 8, "Start": 0.0, "End": 1.0}],
                duration_ms=1000,
            )
        with self.assertRaises(ValueError):
            normalize_nemotron_turns(
                [{"Speaker": 0, "Start": 0.0, "End": 5.0}],
                duration_ms=1000,
            )

    def test_paired_diarization_is_permutation_invariant(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case = prepare_diarization(root)
            report = compare_diarization(
                [case], root, nemotron=DummyDiarizer()
            )
            self.assertEqual(report["paired_completed"], 1)
            for model in ("camplus_live", "nemotron3"):
                row = report["models"][model]
                self.assertEqual(row["summary"]["reference_weighted_der"], 0)
                self.assertEqual(row["summary"]["completed"], 1)
            # The existing replay does not measure isolated CAM++ latency.
            self.assertIsNone(
                report["models"]["camplus_live"]["summary"]["mean_inference_ms"]
            )

    def test_missing_or_invalid_gold_and_checksum_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case = prepare_diarization(root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verified_path(root, {**case, "sha256": "0" * 64})
            broken = dict(case)
            broken["reference_turns"] = []
            with self.assertRaisesRegex(ValueError, "reference"):
                compare_diarization([broken], root, nemotron=DummyDiarizer())
            broken = dict(case)
            broken["reference_turns"] = [
                {"start_ms": 0, "end_ms": 4000, "speaker": "alice"}
            ]
            with self.assertRaisesRegex(ValueError, "human-annotated"):
                compare_diarization([broken], root, nemotron=DummyDiarizer())
            with self.assertRaisesRegex(ValueError, "local WAV"):
                verified_path(root, {
                    "audio": "../escape.wav", "sha256": case["sha256"]
                })

    def test_failed_candidate_cannot_pass_quality_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case = prepare_diarization(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": MANIFEST_SCHEMA,
                "diarization_cases": [case],
            }), encoding="utf-8")
            result = run(
                manifest, root, "diarization", nemotron=FailedDiarizer()
            )
            self.assertFalse(complete(result))
            self.assertEqual(
                result["diarization"]["models"]["nemotron3"]["summary"]["failed"],
                1,
            )

    def test_verification_auc_eer_and_declared_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = write_wave(root, "audio/a.wav", 0.1)
            b = write_wave(root, "audio/b.wav", 0.2)
            c = write_wave(root, "audio/c.wav", -0.3)
            cases = [
                {"id": "positive", "left": a, "right": b,
                 "same_speaker": True},
                {"id": "negative", "left": a, "right": c,
                 "same_speaker": False},
            ]
            report = compare_verification(
                cases, root,
                backends={"camplus": DummyEmbedder(),
                          "ecapa": DummyEmbedder()},
                thresholds={"camplus": 0.5, "ecapa": 0.5},
            )
            for model in report["models"].values():
                self.assertEqual(model["completed"], 2)
                self.assertEqual(model["metrics"]["auc"], 1)
                self.assertEqual(model["metrics"]["eer"], 0)
                self.assertEqual(model["metrics"]["far"], 0)
                self.assertEqual(model["metrics"]["frr"], 0)

    def test_unbalanced_pairs_never_create_fabricated_eer(self):
        result = verification_metrics([(True, 0.9)], threshold=None)
        self.assertIsNone(result["eer"])
        self.assertIsNone(result["auc"])

    def test_merged_verification_requires_same_frozen_manifest(self):
        from tools.speaker_stage1_compare import SCHEMA
        from tools.speaker_stage1_merge import merge_verification

        def artifact(name, digest="a" * 64, passed=True):
            return {
                "schema": SCHEMA,
                "manifest_sha256": digest,
                "reference_status": "operator_supplied_not_independently_verified",
                "verification": {
                    "pair_count": 2,
                    "models": {
                        name: {
                            "requested": 2,
                            "completed": 2 if passed else 1,
                            "failed": 0 if passed else 1,
                            "metrics": {"eer": 0.1},
                        },
                    },
                },
            }

        merged = merge_verification(artifact("camplus"), artifact("ecapa"))
        self.assertTrue(merged["paired_comparison_complete"])
        self.assertTrue(merged["candidate_models_not_promoted"])
        with self.assertRaisesRegex(ValueError, "different"):
            merge_verification(
                artifact("camplus"), artifact("ecapa", digest="b" * 64),
            )
        incomplete = merge_verification(
            artifact("camplus"), artifact("ecapa", passed=False),
        )
        self.assertFalse(incomplete["paired_comparison_complete"])
        with self.assertRaisesRegex(ValueError, "CAM\\+\\+"):
            merge_verification(artifact("ecapa"), artifact("ecapa"))

    def test_nemotron_numpy_scalars_are_accepted(self):
        values = normalize_nemotron_turns(
            [{"Speaker": np.int64(3), "Start": np.float32(0.0),
              "End": np.float32(1.0)}],
            duration_ms=1000,
        )
        self.assertEqual(values[0]["speaker_id"], 4)

    def test_manifest_rejects_duplicate_and_missing_references(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps({
                "schema": MANIFEST_SCHEMA,
                "diarization_cases": [
                    {"id": "x"}, {"id": "x"},
                ],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()

"""Privacy and provenance regression for Fixed195 separated-channel trial builder.

Synthetic constant waveforms prove file validation and trial pairing only.
These tests DO NOT validate real voiceprint accuracy.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from tools.speaker_stage1_pairs import (
    SOURCE_SCHEMA, build, choose_windows, read_sources,
)
from tools.speaker_stage1_compare import load_manifest, verified_path


def put_wav(path: Path, amplitude: float = 0.18, seconds: int = 25):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(
            np.full(
                seconds * 16000,
                int(amplitude * 32767),
                dtype="<i2",
            ).tobytes()
        )
    return sha256(path.read_bytes()).hexdigest()


def fixture(root: Path, *, silent_speaker: bool = False) -> Path:
    names = [
        "fixed195-en-m01.wav",
        "fixed195-en-m01_speaker1.wav",
        "fixed195-en-m01_speaker2.wav",
    ]
    hashes = [
        put_wav(root / "audio" / name,
                0 if silent_speaker and i == 2 else 0.1 + 0.02 * i)
        for i, name in enumerate(names)
    ]
    row = {
        "id": "fixed195-en-m01",
        "kind": "multi",
        "scenario": "authentic_stereo_human_conversation",
        "source_repo": "test/nonproduction",
        "reference_status": "missing_human_transcript",
        "is_scoreable_diarization": False,
        "audio": "audio/" + names[0],
        "sha256": hashes[0],
        "speaker_channels": ["left", "right"],
        "isolated_audio": [
            {"speaker": "speaker_1", "audio": names[1],
             "sha256": hashes[1]},
            {"speaker": "speaker_2", "audio": names[2],
             "sha256": hashes[2]},
        ],
    }
    path = root / "public_multi_unscored.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


class SpeakerStageOnePairTests(unittest.TestCase):
    def test_generated_pairs_are_balanced_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = fixture(root)
            dest = root / "speaker_stage1_manifest.json"
            doc = build(root, source, dest)
            self.assertEqual(len(doc["verification_pairs"]), 4)
            self.assertEqual(
                [p["same_speaker"] for p in doc["verification_pairs"]],
                [True, True, False, False],
            )
            self.assertEqual(doc["diarization_cases"], [])
            self.assertFalse(
                doc["verification_provenance"]["cross_session_identity_evaluable"]
            )
            self.assertTrue(
                doc["verification_provenance"]["requires_manual_stem_review"]
            )
            self.assertEqual(load_manifest(dest)["schema"], doc["schema"])
            self.assertEqual(
                doc["verification_provenance"]["paired_recordings"], 1
            )
            for pair in doc["verification_pairs"]:
                for side in ("left", "right"):
                    verified_path(root, pair[side])
            # A same-person pair uses two disjoint windows, not the same WAV.
            positive = doc["verification_pairs"][0]
            self.assertNotEqual(positive["left"]["audio"],
                                positive["right"]["audio"])
            with self.assertRaises(FileExistsError):
                build(root, source, dest)

    def test_rejects_tampered_stem_or_missing_public_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = fixture(root)
            stem = root / "audio" / "fixed195-en-m01_speaker1.wav"
            stem.write_bytes(stem.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                build(root, source, root / "manifest.json")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                read_sources(root / "not_found.jsonl", root, None)

    def test_comparison_reports_keep_weak_same_session_provenance(self):
        from tools.speaker_stage1_compare import run
        from tools.speaker_stage1_merge import merge_verification

        class SameAudioEmbedding:
            model_id = "fake/unit"
            def embed(self, audio):
                vector = np.zeros(192, dtype=np.float32)
                vector[0] = 1.0
                return vector

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = fixture(root)
            manifest = root / "stage1.json"
            document = build(root, source, manifest)
            cam = run(manifest, root, "verification", backends={
                "camplus": SameAudioEmbedding()
            })
            ecapa = run(manifest, root, "verification", backends={
                "ecapa": SameAudioEmbedding()
            })
            self.assertEqual(
                cam["verification_provenance"],
                document["verification_provenance"]
            )
            merged = merge_verification(cam, ecapa)
            self.assertFalse(
                merged["verification_provenance"]["cross_session_identity_evaluable"]
            )
            ecapa["verification_provenance"] = {
                "kind": "independent_sessions"
            }
            with self.assertRaisesRegex(ValueError, "provenance"):
                merge_verification(cam, ecapa)

    def test_inactive_channels_do_not_fabricate_trials(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = fixture(root, silent_speaker=True)
            with self.assertRaisesRegex(ValueError, "sufficiently active"):
                build(root, source, root / "trials.json")
            self.assertFalse((root / "trials.json").exists())
            self.assertFalse(
                (root / "speaker_stage1_samples").exists()
            )

    def test_rejects_spoofed_diarization_ground_truth(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = fixture(root)
            entry = json.loads(source.read_text(encoding="utf-8"))
            entry["is_scoreable_diarization"] = True
            source.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                build(root, source, root / "trials.json")

    def test_windows_require_usable_and_separate_activity(self):
        silence = np.zeros(16000 * 25, dtype="<i2")
        self.assertEqual(choose_windows(silence), [])
        active = np.full(16000 * 25, 8000, dtype="<i2")
        parts = choose_windows(active)
        self.assertEqual(len(parts), 2)
        self.assertGreaterEqual(parts[1][0] - parts[0][0], 10)
        self.assertEqual(len(parts[0][1]), 5 * 16000)


if __name__ == "__main__":
    unittest.main()

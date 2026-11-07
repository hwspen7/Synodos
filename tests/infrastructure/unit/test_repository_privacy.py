"""Ensure runtime audio, local data and model artifacts never enter Git."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_PREFIXES = (
    "benchmark/",
    "recordings/",
    "meetings/",
    "speakers/",
    "hotwords/",
    "models_cache/",
    "pretrained_models/",
    "data/",
    "datasets/",
    "artifacts/",
    "reports/",
    "results/",
)
FORBIDDEN_SUFFIXES = (
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4",
    ".pcm", ".aac", ".opus", ".aiff", ".aif", ".wma", ".mkv",
    ".mov", ".avi", ".pt", ".pth", ".onnx", ".safetensors",
    ".ckpt", ".gguf", ".pkl", ".pickle", ".npy", ".npz",
    ".sqlite", ".sqlite3", ".db", ".parquet", ".arrow",
    ".jsonl", ".csv", ".tsv", ".zip", ".7z", ".tar", ".tgz",
    ".pem", ".key", ".p12", ".pfx",
)
FORBIDDEN_NAMES = (".env",)


class RepositoryPrivacyTests(unittest.TestCase):
    def test_no_local_audio_data_or_models_are_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        names = result.stdout.decode("utf-8").split("\0")
        disallowed = sorted(
            path for path in names if path and (
                path.startswith(FORBIDDEN_PREFIXES)
                or path.lower().endswith(FORBIDDEN_SUFFIXES)
                or Path(path).name in FORBIDDEN_NAMES
            )
        )
        self.assertEqual(
            disallowed, [],
            "Private runtime, benchmark or model artifacts must remain untracked.",
        )

    def test_ignore_rules_cover_common_local_artifacts(self):
        examples = (
            ".env",
            "recordings/session.wav",
            "meetings/transcript.json",
            "speakers/speakers.pkl",
            "benchmark/fixed195/audio/example.wav",
            "benchmark/fixed195/transcripts/human.json",
            "benchmark/fixed150/audio/example.wav",
            "data/meeting_raw.json",
            "datasets/evaluation.csv",
            "reports/speaker_scores.json",
            "models_cache/model.gguf",
            "checkpoints/model.safetensors",
            "sample.opus",
            "analysis.jsonl",
        )
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--", *examples],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        ignored = set(result.stdout.decode("utf-8").split("\0"))
        self.assertTrue(set(examples) <= ignored)


if __name__ == "__main__":
    unittest.main()

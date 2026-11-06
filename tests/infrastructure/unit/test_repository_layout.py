"""Regression checks for the committed source-document and manifest layout."""
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = (
    "intelligence",
    "speaker-ecapa",
    "speaker-nemotron",
    "speech-addons",
    "speech-dual-vad",
    "speech-whisper",
)


class RepositoryLayoutTests(unittest.TestCase):
    def test_dependency_manifests_live_in_one_directory(self):
        for name in MANIFESTS:
            with self.subTest(name=name):
                self.assertTrue((ROOT / "requirements" / f"{name}.txt").is_file())
                self.assertFalse((ROOT / f"requirements-{name}.txt").exists())

    def test_only_root_readme_is_retained(self):
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertFalse((ROOT / "docs").exists())
        self.assertFalse((ROOT / "PROJECT_CONTEXT.md").exists())
        self.assertFalse((ROOT / "ops" / "README.md").exists())

        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ) if (ROOT / ".git").exists() else None
        if result is not None and result.returncode == 0:
            markdown = sorted(
                item for item in result.stdout.decode("utf-8").split("\0")
                if item.lower().endswith(".md")
            )
            self.assertEqual(markdown, ["README.md"])


if __name__ == "__main__":
    unittest.main()

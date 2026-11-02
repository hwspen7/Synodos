"""Ensure tracked source and documentation contain no CJK text or fullwidth punctuation."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".vue", ".ts", ".css", ".md", ".html",
    ".json", ".toml", ".yaml", ".yml", ".svg", ".txt",
}
TEXT_NAMES = {".gitignore", ".env.example", ".env.template"}
CJK_CHARACTERS = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]")


class EnglishOnlySourceTests(unittest.TestCase):
    def test_no_cjk_characters_in_tracked_text_files(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        failures: list[str] = []
        for path_bytes in result.stdout.split(b"\0"):
            if not path_bytes:
                continue
            relative_path = Path(path_bytes.decode("utf-8"))
            if relative_path.suffix.lower() not in TEXT_SUFFIXES and relative_path.name not in TEXT_NAMES:
                continue
            file_path = ROOT / relative_path
            if not file_path.is_file():
                continue
            content = file_path.read_text(encoding="utf-8")
            for line_number, line in enumerate(content.splitlines(), 1):
                match = CJK_CHARACTERS.search(line)
                if match is not None:
                    failures.append(
                        f"{relative_path}:{line_number}: U+{ord(match.group()):04X}"
                    )
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()

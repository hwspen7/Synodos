"""Repair two verified scope/compatibility errors in the meeting processor.

Run after pulling feature/meeting-knowledge-quality:
    python tools/repair_speaker_scope.py

The operation is idempotent, checks source anchors, validates Python syntax
and creates a local backup before replacing the original file.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import tempfile


PATCHES = (
    (
        '        import os\n        margin = float(os.getenv("MEETING_PHASE07_SPEAKER_MARGIN", "0"))',
        '        margin = float(os.getenv("MEETING_PHASE07_SPEAKER_MARGIN", "0"))',
    ),
    (
        'available_names = {p["id"]: p["name"] for p in self._participants}',
        'available_names = {p["id"]: p["name"] for p in getattr(self, "_participants", [])}',
    ),
)


def repair(root: Path) -> bool:
    path = root / "sessions" / "processor.py"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find meeting processor: {path}")
    source = path.read_text(encoding="utf-8")
    modified = source
    for before, after in PATCHES:
        if before in modified:
            if modified.count(before) != 1:
                raise RuntimeError("Ambiguous patch target, refusing to modify the processor")
            modified = modified.replace(before, after)
        elif after in modified:
            continue
        else:
            raise RuntimeError("Processor differs from the expected branch; refusing unsafe replacement")
    if modified == source:
        print("Speaker scope repair is already applied")
        return False
    ast.parse(modified, filename=str(path))
    backup = path.with_name("processor.py.before-speaker-scope-repair.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=".processor-scope-", suffix=".py", delete=False,
    ) as temp:
        temp.write(modified)
        temporary = Path(temp.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Applied safe speaker processor scope repair: {path}")
    print(f"Backup saved to: {backup}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    repair(arguments.root.resolve())

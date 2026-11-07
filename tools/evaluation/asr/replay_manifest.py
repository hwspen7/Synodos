"""Build an immutable ten-meeting manifest from completed Nemotron replays.

Reject missing, mismatched or partial GPU tracks. Existing CAM++ replays
are never counted as Nemotron evidence. No audio or raw transcript is copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.evaluation.ami.ami_shadow10_prepare import SELECTED
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.evaluation.ami.ami_nemotron_live_score import _validated_independent


def verified_replay(root: Path, meeting: str, path: Path) -> bool:
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("server_ready", {}).get("independent_diarization") != "nemotron3":
            return False
        _validated_independent(data)
        score_case(root, meeting, path)
        return True
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        return False


def build_manifest(root: Path) -> dict[str, str]:
    selected = {}
    for meeting, _ in SELECTED:
        files = sorted(
            root.glob(f"replay_*{meeting}*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        valid = next((p for p in files[:100]
                      if verified_replay(root, meeting, p)), None)
        if valid is None:
            raise ValueError(
                f"No complete, frozen-reference-matched Nemotron replay "
                f"found for {meeting}. Run the local GPU replay first.")
        selected[meeting] = str(valid.resolve())
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    private = Path("benchmark/fixed195").resolve()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if (not root.is_relative_to(private)
            or not output.is_relative_to(private)
            or output.suffix != ".json" or output.exists()):
        parser.error("Read prepared private AMI and write a NEW private JSON")
    doc = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "validated_replays": len(doc),
        "manifest": str(output),
        "note": "No new model inference or quality improvement was performed.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

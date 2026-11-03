"""Offline speaker and evidence audit of an exported meeting JSON file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase07.evidence import evidence_audit
from phase07.speakers import speaker_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    transcript = data.get("transcript", {})
    segments = transcript.get("segments") or transcript.get("paragraphs") or []
    if not isinstance(segments, list):
        parser.error("Invalid transcript segment list")
    stats = {"speakers": speaker_stats(segments)}
    if args.artifact:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        ids = [row.get("id") for row in segments if isinstance(row, dict)]
        stats["evidence"] = evidence_audit(artifact, ids)
    report = json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()

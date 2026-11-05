"""Export private real meeting ASR segments for independent audio annotation.

This does not generate gold truth or mark any meeting as reviewed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from tools.phase34_release_contract import sha256_file, write_new


def export_review(args):
    from tools.phase34_full_acceptance import guard_environment
    guard_environment()
    from intelligence import store

    if len(args.meeting_id) != len(args.audio_path) or len(args.meeting_id) < 3:
        raise ValueError("Provide at least three paired --meeting-id and --audio-path values")
    if len(set(args.meeting_id)) != len(args.meeting_id):
        raise ValueError("Meeting IDs must be unique")
    if args.manifest.exists() or args.csv.exists():
        raise ValueError("Refusing to overwrite review artifacts")
    rows = []
    meetings = []
    for mid, audio in zip(args.meeting_id, args.audio_path):
        if not audio.is_file():
            raise ValueError("Audio missing: " + str(audio))
        revision, segments = store.meeting_segments(mid)
        if not segments:
            raise ValueError("Meeting missing saved segments: " + mid)
        meetings.append({"meeting_id": mid,
                         "audio_file": str(audio.resolve()),
                         "audio_sha256": sha256_file(audio),
                         "audio_reviewed": False,
                         "synthetic_audio": False,
                         "revision_at_export": revision})
        for s in segments:
            rows.append({"meeting_id": mid, "segment_id": s["id"],
                         "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                         "speaker": s["speaker"], "language": s["language"],
                         "asr_text_not_audio_gold": s["text"],
                         "manual_audio_check": "UNREVIEWED"})
    draft = {"schema": "meeting-phase34-real-audio-gold-v1",
             "corpus_type": "real_recorded_meetings",
             "gold_source": "independent_human_audio_review",
             "reviewer_id": "", "meetings": meetings, "cases": []}
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(args.manifest, draft)
    print("DRAFT ONLY: audio labels, timestamps and 26+ human reviewed cases are required.")
    print("PRIVATE REVIEW SHEET:", args.csv)
    print("UNREVIEWED MANIFEST:", args.manifest)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meeting-id", action="append", required=True)
    p.add_argument("--audio-path", type=Path, action="append", required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        return export_review(args)
    except Exception as exc:
        print("GOLD EXPORT BLOCKED:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

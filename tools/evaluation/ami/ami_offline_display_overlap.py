"""Re-score canonical versus actually displayed overlapping-voice activity.

Only reads immutable AMI replay and frozen source proof. Does not run
Nemotron/Whisper, rewrite transcript, invent words or alter DER acceptance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from speech.diarization.overlap_content_risk import overlap_content_risk
from tools.evaluation.ami.ami_nemotron_live_score import _validated_independent
from tools.evaluation.ami.ami_nemotron_reference_diagnostic import (
    display_projection_turns,
)
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.evaluation.speakers.speaker_stage1_compare import baseline_turns


def offline_display_overlap_report(
    root: Path, meeting: str, replay_path: Path,
) -> dict:
    # The frozen audio/reference and replay provenance must match before
    # making any attribution statement about this test case.
    source = score_case(root, meeting, replay_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    acoustic = _validated_independent(replay)
    original = baseline_turns(replay, duration_ms=240000)
    displayed, split_count = display_projection_turns(replay, duration_ms=240000)
    raw = overlap_content_risk(acoustic, original)
    ui = overlap_content_risk(acoustic, displayed)
    logged = replay.get("content_overlap_audit")
    if isinstance(logged, dict) and (
        logged.get("overlap_acoustic_clock_ms") != raw["overlap_acoustic_clock_ms"]
        or logged.get("overlap_clock_underrepresented_ms")
        != raw["overlap_clock_underrepresented_ms"]
    ):
        raise ValueError(
            "Offline canonical overlap audit disagrees with recorded replay; "
            "investigate origins and source identity before comparing."
        )
    return {
        "schema": "meeting-ami-offline-displayed-overlap-v1",
        "meeting": meeting,
        "source_audio_sha256": source["source_audio_sha256"],
        "frozen_original_reference_sha256": source["reference_source_sha256"],
        "replay_sha256": sha256(replay_path.read_bytes()).hexdigest(),
        "source_complete": True,
        "display_speaker_cue_splits": split_count,
        "canonical_caption_activity": raw,
        "actual_ui_display_activity": ui,
        "difference_in_underrepresented_acoustic_clock_ms": (
            ui["overlap_clock_underrepresented_ms"]
            - raw["overlap_clock_underrepresented_ms"]
        ),
        "note": (
            "Reference-free diarization activity only; not recovered words, "
            "not an independent second-speaker transcript, not WER/CER, "
            "not canonical published-caption DER. A zero time delta here "
            "does not establish that source-separated speech is decoded."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--meeting", default="TS3003b")
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, replay, output = (
        args.root.resolve(), args.replay.resolve(), args.output.resolve(),
    )
    private = Path("benchmark/fixed195").resolve()
    if (not root.is_relative_to(private)
            or not replay.is_relative_to(root) or not replay.is_file()
            or not output.is_relative_to(root) or output.exists()
            or output.suffix.lower() != ".json"):
        parser.error("Use existing frozen E-drive replay and new local JSON output")
    report = offline_display_overlap_report(root, args.meeting, replay)
    with output.open("x", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

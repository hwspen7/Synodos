"""Sequential, reproducible 10-meeting AMI live replay and DER audit.

Must be run from the normal backend environment while a local live
WebSocket server is already running. It never downloads data or alters
backend models, and does not assert quality from unit tests alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from tools.ami_shadow10_prepare import SELECTED, checksum
from tools.ami_shadow10_score import score_case

SCHEMA = "meeting-ami-shadow10-batch-live-evaluation-v1"


def aggregate(results: list[dict]) -> dict:
    """Weight speaker-time DER by AMI reference speaker seconds per clip."""
    if not results:
        raise ValueError("No validated cases")
    denominator = sum(x["metrics"]["reference_speaker_seconds"] for x in results)
    if denominator <= 0:
        raise ValueError("No reference speaker time")
    fields = (
        "der", "missed_speaker_time", "false_alarm_time", "confusion_time",
    )
    return {
        "case_count": len(results),
        "total_reference_speaker_seconds": round(denominator, 2),
        "weighted_metrics": {
            field: round(sum(
                x["metrics"][field] * x["metrics"]["reference_speaker_seconds"]
                for x in results
            ) / denominator, 5)
            for field in fields
        },
        "sum_unidentified_segments": sum(
            x["unidentified"]["count"] for x in results
        ),
        "reference_max_speaker_count": max(
            x["metrics"]["reference_speaker_count"] for x in results
        ),
        "reported": [
            {
                "meeting": x["meeting"],
                "der": x["metrics"]["der"],
                "reference_speaker_count": x["metrics"]["reference_speaker_count"],
                "predicted_speaker_count": x["metrics"]["predicted_speaker_count"],
                "unidentified": x["unidentified"]["count"],
                "reference_speech_without_captions_s": (
                    x["reference_speech_without_captions"][
                        "reference_speech_without_captions_seconds"
                    ]
                ),
            }
            for x in results
        ],
        "note": (
            "Reference-time weighted approximate DER from separately scored "
            "AMI meetings. This does not establish cross-meeting voice "
            "identity correctness, unknown-speaker FAR or a Zoom comparison."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark/fixed195/ami_shadow10"),
    )
    parser.add_argument("--tag", default="cluster_bootstrap_01")
    parser.add_argument("--case", choices=[m for m, _ in SELECTED])
    parser.add_argument("--resume", action="store_true",
                        help="Validate existing replay, do not overwrite it")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", args.tag):
        parser.error("--tag must be 1-32 alphanumeric/hyphen characters")
    if not 360 <= args.timeout_seconds <= 3600:
        parser.error("timeout must be 360..3600 seconds")
    root = args.root.resolve()
    if not root.is_relative_to(Path("benchmark/fixed195").resolve()):
        parser.error("AMI batch stays in private gitignored Fixed195 directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        parser.error("Run tools.ami_shadow10_prepare to download AMI excerpts first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        parser.error("AMI manifest is incomplete; download all ten cases first")
    all_cases = {case["source_meeting"]: case for case in manifest["cases"]}
    if set(all_cases) != {m for m, _ in SELECTED}:
        parser.error("AMI manifest does not contain the expected ten meetings")
    picked = [m for m, _ in SELECTED if args.case is None or m == args.case]
    results = []
    for meeting in picked:
        case = all_cases[meeting]
        audio = (root / case["audio"]).resolve()
        if (not audio.is_relative_to(root) or not audio.is_file()
                or checksum(audio) != case.get("sha256")):
            raise ValueError("AMI source has changed for " + meeting)
        replay = root / f"replay_{meeting}_{args.tag}.json"
        audited = root / f"audit_{meeting}_{args.tag}.json"
        if replay.exists() and not args.resume:
            raise FileExistsError(
                "Refusing to overwrite a prior live replay: " + str(replay)
            )
        if not replay.exists():
            process = subprocess.run([
                sys.executable, "-m", "tools.live_caption_ws_replay",
                "--audio", str(audio), "--loops", "1", "--speed", "1.0",
                "--output", str(replay),
            ], cwd=str(Path(__file__).resolve().parents[1]),
                timeout=args.timeout_seconds, check=False,
                capture_output=False)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Live replay failed for {meeting}; no accuracy claim"
                )
        result = score_case(root, meeting, replay)
        if audited.exists():
            if not args.resume:
                raise FileExistsError(
                    "Refusing to overwrite prior AMI audit: " + str(audited)
                )
            old = json.loads(audited.read_text(encoding="utf-8"))
            if (old.get("source_audio_sha256") != result["source_audio_sha256"]
                    or old.get("replay_sha256") != result["replay_sha256"]):
                raise ValueError("Existing audit provenance mismatch")
        else:
            with audited.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
        results.append(result)
        print(f"{meeting}: DER={result['metrics']['der']:.5f} "
              f"speakers={result['metrics']['predicted_speaker_count']}/"
              f"{result['metrics']['reference_speaker_count']}", flush=True)
    summary = aggregate(results)
    summary["schema"] = SCHEMA
    summary["tag"] = args.tag
    destination = root / f"summary_{args.tag}_{'all' if args.case is None else args.case}.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite prior AMI summary")
    with destination.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "summary": str(destination),
        "case_count": summary["case_count"],
        "weighted_metrics": summary["weighted_metrics"],
        "sum_unidentified_segments": summary["sum_unidentified_segments"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

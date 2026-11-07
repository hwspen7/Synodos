"""Download and verify AMI close-talking headset channels without assuming speakers.

Recorded real audio and original NXT text remain separate until microphone
to speaker assignment and audibility are independently checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import wave

from tools.evaluation.corpora.prepare_meeting_reference_corpus import MEETINGS, sha256_file
from tools.evaluation.corpora.prepare_real_meeting_evaluation import DEFAULT_ROOT, check_root, source_audio

OFFICIAL_PREFIX = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
SCHEMA = "meeting-ami-individual-headsets-v1"
CHANNELS = range(4)
MAX_BYTES = 140 * 1024 * 1024


def official_url(meeting: str, channel: int) -> str:
    if meeting not in MEETINGS or type(channel) is not int or channel not in CHANNELS:
        raise ValueError("Only three frozen AMI recordings and headset channels 0-3")
    return (OFFICIAL_PREFIX + "/" + meeting + "/audio/" +
            meeting + ".Headset-" + str(channel) + ".wav")


def verify_channel(path: Path, meeting_duration: float) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Missing or oversized WAV headset channel")
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(),
                reader.getsampwidth(), reader.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Headset must be 16 kHz mono PCM16")
        duration = reader.getnframes() / reader.getframerate()
        if abs(duration - meeting_duration) > 5:
            raise ValueError("Headset recording duration mismatches source meeting")
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "duration_seconds": round(duration, 3),
        "sample_rate": 16000, "channels": 1,
    }


def download_one(url: str, destination: Path):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".download")
    if temp.exists():
        raise FileExistsError("Interrupted temporary download exists; inspect before retry")
    try:
        request = urllib.request.Request(url, headers={
            "User-Agent": "Meeting-Reference-Integrity/1.0",
        })
        with urllib.request.urlopen(request, timeout=60) as response, temp.open("xb") as out:
            length = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                length += len(chunk)
                if length > MAX_BYTES:
                    raise ValueError("Official WAV exceeds download size cap")
                out.write(chunk)
        if length < 128 * 1024:
            raise ValueError("Unexpectedly small headset WAV")
        temp.rename(destination)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def prepare(root: Path, *, apply: bool) -> tuple[dict, Path]:
    root = check_root(root)
    inventory = root / "ihm" / "individual_headset_inventory.json"
    if inventory.exists():
        raise FileExistsError("Existing source inventory cannot be overwritten")
    meetings = {}
    for meeting in MEETINGS:
        _, mix_duration = source_audio(root, meeting)
        channels = {}
        for channel in CHANNELS:
            url = official_url(meeting, channel)
            file = root / "ihm" / meeting / (meeting + ".Headset-" +
                                               str(channel) + ".wav")
            if apply:
                download_one(url, file)
                verified = verify_channel(file, mix_duration)
            else:
                verified = {"file": str(file), "verification_pending": True}
            channels[str(channel)] = {
                **verified, "official_url": url,
                "speaker_assignment_verified": False,
                "human_audibility_checked": False,
                "reference_text_used_for_training": False,
            }
            print("IHM", meeting, channel,
                  "VERIFIED" if apply else "PLANNED", flush=True)
        meetings[meeting] = {
            "original_mix_duration_seconds": round(mix_duration, 3),
            "channels": channels,
            "speaker_channel_mapping_established": False,
        }
    report = {
        "schema": SCHEMA, "source": "AMI Meeting Corpus",
        "source_license": "CC BY 4.0",
        "recordings": meetings, "headset_channel_count": len(MEETINGS) * 4,
        "human_reviewed_reference_available": False,
        "eligible_for_model_scoring": False,
        "may_auto_promote": False,
        "note": "Channel index is not asserted to equal speaker letter. Verify microphone-to-speaker mapping and audibility separately.",
    }
    if apply:
        inventory.parent.mkdir(parents=True, exist_ok=True)
        with inventory.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=True, indent=2)
            output.write("\n")
    return report, inventory


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--apply", action="store_true",
                   help="Explicitly download 12 separate official headset recordings")
    args = p.parse_args(argv)
    try:
        result, path = prepare(args.root, apply=args.apply)
        print("INDIVIDUAL AMI HEADSETS", result["headset_channel_count"],
              "channels", "INVENTORY" if args.apply else "DRY RUN", path)
        print("SPEAKER MAPPING UNVERIFIED. HUMAN GOLD NOT_READY.")
        return 0
    except (OSError, ValueError, TypeError, KeyError, wave.Error) as exc:
        print("IHM PREPARATION BLOCKED", type(exc).__name__, str(exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

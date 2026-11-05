"""Acquire and verify public AMI audio and annotations for independent review.

Never imports recordings into a database, invents human labels, or changes models.
All media stays outside the Git repository in the caller-selected directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import wave
import zipfile
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/resolve/main/"
ASSETS = (
    ("sdm/EN2002a.Mix-Headset.wav", "a15017d0d3c871866971ee178052a0ce3a7e44d6fbf89fc23dcab326511b65f5"),
    ("sdm/ES2004a.Mix-Headset.wav", "3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f"),
    ("sdm/IS1009a.Mix-Headset.wav", "6eb5a0ede0d9e72794f976ce7bea5b78133eae969f99b4c5418b43c2468d25b1"),
    ("annotations/ami_public_manual_1.6.2.zip", "56e29112cd89cfba76fbf8f10215eb245cc89b0d3d27355ba14b406a7ea7776c"),
)
MEETINGS = ("EN2002a", "ES2004a", "IS1009a")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_file(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    found = sha256_file(path)
    if found != expected:
        raise ValueError(f"Existing file has an unexpected SHA256; refusing to replace: {path}")
    return True


def download_asset(relative: str, expected: str, root: Path) -> Path:
    path = root / relative
    if verified_file(path, expected):
        print(f"VERIFIED EXISTING {relative}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial-{os.getpid()}")
    if tmp.exists():
        raise ValueError(f"Temporary download already exists: {tmp}")
    try:
        req = Request(BASE + relative, headers={"User-Agent": "MeetingReferenceCorpus/1.0"})
        with urlopen(req, timeout=90) as response, tmp.open("xb") as stream:
            if response.url.split("/", 3)[0].lower() != "https:":
                raise ValueError("Download redirected to a non-HTTPS URL")
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if sha256_file(tmp) != expected:
            raise ValueError(f"Downloaded content SHA256 mismatch: {relative}")
        if path.exists():
            raise ValueError(f"Target appeared during download; refusing overwrite: {path}")
        tmp.rename(path)
        print(f"DOWNLOADED AND VERIFIED {relative}")
        return path
    finally:
        tmp.unlink(missing_ok=True)


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as stream:
        sr = stream.getframerate()
        frames = stream.getnframes()
        channels = stream.getnchannels()
        duration = frames / sr if sr else 0.0
        if duration < 300 or channels < 1 or sr < 8000:
            raise ValueError(f"Unexpected meeting audio characteristics: {path}")
        return {"sample_rate": sr, "channels": channels, "duration_seconds": round(duration, 2),
                "frames": frames, "sample_width_bytes": stream.getsampwidth()}


def inspect_annotations(path: Path) -> dict:
    counts = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        for meeting in MEETINGS:
            words = [name for name in names if name.replace("\\", "/").split("/")[-1].startswith(meeting + ".")
                     and name.endswith(".words.xml")]
            segments = [name for name in names if name.replace("\\", "/").split("/")[-1].startswith(meeting + ".")
                        and name.endswith(".segments.xml")]
            if not words or not segments:
                raise ValueError(f"Missing original manual words or segments for {meeting}")
            counts[meeting] = {"word_xml_files": len(words), "segment_xml_files": len(segments)}
    return counts


def run(root: Path, download: bool) -> dict:
    root = root.resolve()
    if os.name == "nt" and root.drive.lower() != "e:":
        raise ValueError("Use the E drive for this corpus, not the system drive")
    if download:
        root.mkdir(parents=True, exist_ok=True)
    for relative, expected in ASSETS:
        path = root / relative
        if download:
            download_asset(relative, expected, root)
        elif not verified_file(path, expected):
            raise FileNotFoundError(f"Not downloaded: {path}. Run with --download to acquire it")
    report = {"corpus": "AMI Meeting Corpus", "source": BASE,
              "license": "CC BY 4.0 (AMI Consortium)", "meetings": {},
              "annotation_state": "REFERENCE_ONLY_NOT_INDEPENDENTLY_AUDIO_REVIEWED",
              "meeting_test_imported": False, "human_gold_created": False}
    for meeting in MEETINGS:
        relative = f"sdm/{meeting}.Mix-Headset.wav"
        path = root / relative
        report["meetings"][meeting] = {"relative_audio_file": relative,
                                       "sha256": dict(ASSETS)[relative], **inspect_wav(path)}
    report["manual_annotations"] = inspect_annotations(root / ASSETS[-1][0])
    receipt = root / "reference_inventory.json"
    receipt.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("REFERENCE ASSETS VERIFIED (NOT HUMAN-REVIEWED GOLD)")
    print("INVENTORY:", receipt)
    for mid, item in report["meetings"].items():
        print(mid, item["duration_seconds"], "seconds", item["sample_rate"], "Hz")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\AI-Cache\MeetingReference\AMI"))
    parser.add_argument("--download", action="store_true", help="Explicitly download public assets; otherwise verify only")
    args = parser.parse_args(argv)
    try:
        run(args.root, args.download)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        print("REFERENCE CORPUS NOT_READY:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

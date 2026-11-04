"""Prepare ten real 4-minute AMI meetings for ECAPA shadow replay.

All audio, timestamps and manifests remain under the gitignored Fixed195 root.
The audio comes from the AMI Corpus (CC BY 4.0); RTTM labels come from the
pyannote AMI diarization setup's human-annotation-derived 'only_words' split.

The selected spans were screened against pinned RTTM references for repeated
contributions by >=2 speakers. No model output is used as reference.
Run only on a machine permitted to download the AMI Corpus.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import urllib.request
import wave

ANNOTATION_COMMIT = "67c2d539286e89f68952d5dcf83912bd9f01dfae"
ANNOTATION_BASE = (
    "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/"
    + ANNOTATION_COMMIT + "/only_words/rttms/test/"
)
AMI_AUDIO_BASE = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "HeadsetAudio/"
)
AMI_BACKUP_BASE = (
    "https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/"
    "resolve/main/sdm/"
)
SAMPLE_RATE = 16000
DURATION_S = 240
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
SCHEMA = "meeting-ami-shadow10-human-referenced-v1"

# Ten real recordings from four distinct AMI session groups.
# Every 240-second span has several substantial, repeated speaking turns,
# verified using the annotation commit pinned above.
SELECTED = (
    ("EN2002a", 1830),
    ("EN2002b", 300),
    ("EN2002d", 1185),
    ("ES2004a", 555),
    ("ES2004c", 1245),
    ("ES2004d", 1395),
    ("IS1009b", 855),
    ("IS1009d", 570),
    ("TS3003b", 1500),
    ("TS3003c", 1440),
)


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path) -> None:
    """Download exclusively to the selected E-drive work directory.

    Never use the OS temp folder or override existing source bytes. A failed
    download leaves no reusable partial file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    temp = output.with_name(output.name + ".part")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={
                "User-Agent": "Meeting-AMI-Shadow10/1.0"
            }), timeout=120,
        ) as response, temp.open("xb") as dest:
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Source download exceeds 200 MiB limit")
                dest.write(block)
            if count < 100:
                raise ValueError("Downloaded source is unexpectedly small")
        temp.rename(output)
    finally:
        temp.unlink(missing_ok=True)


def download_rttm(folder: Path, meeting: str) -> Path:
    path = folder / "rttm" / (meeting + ".rttm")
    _download(ANNOTATION_BASE + meeting + ".rttm", path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("RTTM source is unexpectedly large")
    return path


def parse_reference(raw: str, meeting: str, offset: int) -> list[dict]:
    """Crop human-derived RTTM to local clip time without losing overlap."""
    result = []
    duration_ms = DURATION_S * 1000
    for line in raw.splitlines():
        if not line.strip():
            continue
        columns = line.split()
        if (len(columns) != 10 or columns[0] != "SPEAKER"
                or columns[1] != meeting):
            raise ValueError("Unexpected AMI RTTM annotation")
        speaker = columns[7]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", speaker):
            raise ValueError("Malformed AMI speaker ID")
        try:
            start = float(columns[3]) - offset
            end = start + float(columns[4])
        except ValueError as exc:
            raise ValueError("Invalid AMI RTTM timing") from exc
        left = max(0, round(start * 1000))
        right = min(duration_ms, round(end * 1000))
        if right > left:
            result.append({
                "start_ms": left, "end_ms": right, "speaker": speaker,
            })
    result.sort(key=lambda item: (item["start_ms"], item["speaker"]))
    return result


def validate_reference(turns: list[dict]) -> dict:
    """Require two speakers with multiple clear annotated contributions."""
    speech_ms = {}
    counts = {}
    for turn in turns:
        speaker = turn["speaker"]
        speech_ms[speaker] = speech_ms.get(speaker, 0) + (
            turn["end_ms"] - turn["start_ms"]
        )
        counts[speaker] = counts.get(speaker, 0) + 1
    qualified = [
        label for label in speech_ms
        if speech_ms[label] >= 25000 and counts[label] >= 8
    ]
    if len(qualified) < 2:
        raise ValueError(
            "Selected excerpt lacks two frequently speaking reference speakers"
        )
    return {
        "reference_speaker_count": len(speech_ms),
        "qualified_speaker_count": len(qualified),
        "annotated_turn_count": len(turns),
        "reference_speaker_seconds": round(sum(speech_ms.values()) / 1000, 2),
    }


def crop_pcm16(source: Path, dest: Path, offset: int) -> None:
    """Copy a 240 s 16-kHz mono PCM clip without resampling or model imports."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(),
                reader.getsampwidth(), reader.getcomptype()) != (
                    SAMPLE_RATE, 1, 2, "NONE"
                ):
            raise ValueError(
                "AMI headset mix is not the expected 16 kHz mono PCM16 WAV"
            )
        length = DURATION_S * SAMPLE_RATE
        if reader.getnframes() < (offset * SAMPLE_RATE) + length:
            raise ValueError("AMI recording ends before the selected 240 s span")
        reader.setpos(offset * SAMPLE_RATE)
        data = reader.readframes(length)
        if len(data) != length * 2:
            raise ValueError("Unexpected EOF in AMI recording")
    with dest.open("xb") as output:
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(data)


def _write_json_exclusive(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as dest:
        json.dump(data, dest, ensure_ascii=False, indent=2)
        dest.write("\n")


def prepare_case(root: Path, meeting: str, offset: int,
                 *, keep_full: bool = False,
                 downloader=_download) -> dict:
    if (meeting, offset) not in SELECTED:
        raise ValueError("Only pinned AMI excerpt IDs are accepted")
    audio_relative = f"audio/ami-{meeting}-{offset}-{offset + DURATION_S}.wav"
    audio_path = root / audio_relative
    meta_path = root / "cases" / (meeting + ".json")
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if (data.get("source_meeting") != meeting
                or data.get("start_seconds") != offset
                or data.get("audio") != audio_relative
                or not audio_path.is_file()
                or checksum(audio_path) != data.get("sha256")):
            raise ValueError("Previously prepared AMI case changed")
        validate_reference(data["reference_turns"])
        return data
    if audio_path.exists():
        raise FileExistsError(
            "Untracked output WAV exists. Refusing to overwrite: "
            + str(audio_path)
        )

    rttm = download_rttm(root, meeting)
    turns = parse_reference(rttm.read_text(encoding="utf-8"),
                            meeting, offset)
    stats = validate_reference(turns)
    source_name = meeting + ".Mix-Headset.wav"
    cached = root / "cache" / source_name
    used_url = AMI_AUDIO_BASE + source_name
    if not cached.exists():
        try:
            downloader(used_url, cached)
        except (OSError, ValueError) as exc:
            # Explicit fallback to a verified redistribution of the same
            # 16-meeting AMI evaluation subset, not a random replacement.
            if cached.exists():
                raise
            try:
                used_url = AMI_BACKUP_BASE + source_name
                downloader(used_url, cached)
            except Exception as fallback:
                raise RuntimeError(
                    "AMI source unavailable from official mirror and backup"
                ) from fallback
    try:
        crop_pcm16(cached, audio_path, offset)
        data = {
            "id": f"ami-{meeting}-{offset}",
            "source_meeting": meeting,
            "source_audio_url": AMI_AUDIO_BASE + source_name,
            "source_download_url": used_url,
            "source_audio_sha256": checksum(cached),
            "reference_url": ANNOTATION_BASE + meeting + ".rttm",
            "reference_source_sha256": checksum(rttm),
            "reference_kind": "AMI human-annotation-derived only_words",
            "license": "CC BY 4.0",
            "start_seconds": offset,
            "end_seconds": offset + DURATION_S,
            "duration_seconds": DURATION_S,
            "audio": audio_relative,
            "sha256": checksum(audio_path),
            "reference_turns": turns,
            **stats,
        }
        _write_json_exclusive(meta_path, data)
    except Exception:
        if not meta_path.exists():
            audio_path.unlink(missing_ok=True)
        raise
    finally:
        if not keep_full:
            cached.unlink(missing_ok=True)
    return data


def write_inventory(root: Path) -> dict:
    folder = root / "cases"
    all_cases = []
    for meeting, _ in SELECTED:
        path = folder / (meeting + ".json")
        if path.exists():
            case = json.loads(path.read_text(encoding="utf-8"))
            audio = root / case["audio"]
            if not audio.is_file() or checksum(audio) != case.get("sha256"):
                raise ValueError("AMI inventory audio checksum mismatch")
            all_cases.append(case)
    doc = {
        "schema": SCHEMA,
        "annotation_commit": ANNOTATION_COMMIT,
        "description": (
            "Four-minute real AMI meeting excerpts, with separately "
            "cropped human-derived diarization references. No transcript "
            "text or personalized enrollment samples are included."
        ),
        "cases": all_cases,
        "prepared_count": len(all_cases),
        "expected_count": len(SELECTED),
        "complete": len(all_cases) == len(SELECTED),
    }
    target = root / "manifest.json"
    temp = root / "manifest.json.part"
    if temp.exists():
        raise FileExistsError("Stale AMI inventory temp file")
    _write_json_exclusive(temp, doc)
    os.replace(temp, target)
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("benchmark/fixed195/ami_shadow10")
    )
    parser.add_argument("--case-id", choices=[meeting for meeting, _ in SELECTED],
                        help="Download and prepare one 4-minute sample")
    parser.add_argument("--keep-full", action="store_true",
                        help="Keep original full-meeting WAV files in the E-drive cache")
    args = parser.parse_args()
    root = args.root.resolve()
    # Explicitly refuse output to non-private/unignored paths of the project.
    private = Path("benchmark/fixed195").resolve()
    if not root.is_relative_to(private):
        parser.error("--root must be inside benchmark/fixed195 (gitignored)")
    root.mkdir(parents=True, exist_ok=True)
    picks = [item for item in SELECTED
             if args.case_id is None or item[0] == args.case_id]
    for meeting, offset in picks:
        print(f"Preparing {meeting}, {offset}-{offset + DURATION_S}s", flush=True)
        result = prepare_case(root, meeting, offset,
                              keep_full=args.keep_full)
        print(f"  {result['reference_speaker_count']} reference speakers, "
              f"{result['annotated_turn_count']} annotated turns, "
              f"{result['audio']}", flush=True)
    inventory = write_inventory(root)
    print(json.dumps({
        "manifest": str(root / "manifest.json"),
        "prepared_count": inventory["prepared_count"],
        "complete": inventory["complete"],
        "samples": [x["audio"] for x in inventory["cases"]],
        "reference_note": "AMI only_words annotations may omit vocal sounds",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a private, time-aligned AMI human-word reference for TS3003b.

Source: AMI Corpus manual annotations v1.6.2, CC BY 4.0, NXT words XML.
The human transcript contains overlapping voices that may NOT be audible
in the mixed headset audio. This creates a reproducible reference for
review, not an automatic release approval or an assertion that every
overlapping word must be recoverable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from tools.ami_shadow10_prepare import DURATION_S, SELECTED, checksum

ANNOTATION_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
SCHEMA = "meeting-ami-human-words-reference-v1"
MAX_ZIP_BYTES = 48 * 1024 * 1024
MAX_XML_BYTES = 4 * 1024 * 1024
MEETING = "TS3003b"
START_S = dict(SELECTED)[MEETING]
END_S = START_S + DURATION_S


@dataclass(frozen=True)
class HumanWord:
    speaker: str
    start_s: float
    end_s: float
    text: str


def _tag_name(element: ET.Element) -> str:
    return str(element.tag).split("}", 1)[-1]


def parse_ami_word_xml(
    data: bytes, *, speaker: str, start_s: float = START_S,
    end_s: float = END_S,
) -> list[HumanWord]:
    """Select only lexical <w> entries fully inside the AMI audio crop."""
    if not re.fullmatch(r"[A-Z]", speaker):
        raise ValueError("Unrecognized AMI speaker ID")
    if len(data) > MAX_XML_BYTES or not 0 <= start_s < end_s:
        raise ValueError("Unsafe AMI annotation size or time interval")
    root = ET.fromstring(data)
    result = []
    for element in root.iter():
        if _tag_name(element) != "w":
            continue
        raw = str(element.text or "").strip()
        if not raw or re.fullmatch(r"<[^>]+>", raw):
            continue
        begin_s = float(element.attrib["starttime"])
        finish_s = float(element.attrib["endtime"])
        if (not math.isfinite(begin_s) or not math.isfinite(finish_s)
                or finish_s <= begin_s):
            raise ValueError("Non-finite or invalid AMI word timestamp")
        if begin_s < start_s or finish_s > end_s:
            continue
        result.append(HumanWord(speaker, begin_s, finish_s, raw))
    result.sort(key=lambda w: (w.start_s, w.end_s))
    return result


def words_from_official_archive(
    path: Path, *, meeting: str = MEETING,
    start_s: float = START_S, end_s: float = END_S,
) -> tuple[list[HumanWord], dict]:
    if meeting != MEETING or (start_s, end_s) != (START_S, END_S):
        raise ValueError("Only the frozen TS3003b 1500-1740s excerpt is supported")
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("Unexpected AMI annotation archive size")
    by_speaker = {}
    sources = {}
    with ZipFile(path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            match = re.fullmatch(
                re.escape(meeting) + r"\.([A-Z])\.words\.xml", name,
            )
            if not match:
                continue
            speaker = match.group(1)
            if speaker in by_speaker:
                raise ValueError("Duplicate AMI speaker word file")
            if member.file_size > MAX_XML_BYTES or member.is_dir():
                raise ValueError("Unsafe AMI speaker word file")
            raw = archive.read(member)
            by_speaker[speaker] = parse_ami_word_xml(
                raw, speaker=speaker, start_s=start_s, end_s=end_s,
            )
            sources[name] = sha256(raw).hexdigest()
    if len(by_speaker) < 2:
        raise ValueError("Archive lacks multiple TS3003b speaker word files")
    words = sorted(
        (w for group in by_speaker.values() for w in group),
        key=lambda w: (w.start_s, w.end_s, w.speaker),
    )
    if not words:
        raise ValueError("No time-aligned words in selected AMI clip")
    overlaps = 0
    for index, word in enumerate(words):
        if any(
            other.speaker != word.speaker
            and other.start_s < word.end_s
            and other.end_s > word.start_s
            for other in words[max(0, index - 30):index]
        ):
            overlaps += 1
    return words, {
        "source": "AMI manual annotations v1.6.2 (CC BY 4.0)",
        "source_url": ANNOTATION_URL,
        "archive_sha256": checksum(path),
        "xml_file_sha256": sources,
        "meeting": meeting, "start_seconds": start_s,
        "end_seconds": end_s,
        "word_count": len(words),
        "speaker_word_counts": {
            key: len(rows) for key, rows in sorted(by_speaker.items())
        },
        "overlapping_word_entries": overlaps,
        "overlap_note": (
            "Overlapping lexical annotations are not necessarily intelligible "
            "from the mixed microphone audio. Global CER/WER against this "
            "multi-speaker chronological text is only a screening metric, "
            "not independently sufficient for meeting transcription acceptance."
        ),
    }


def download_official_archive(destination: Path) -> None:
    """Download with a cap to an E-drive private directory, never overwrite."""
    if destination.exists():
        return
    if destination.suffix.lower() != ".zip":
        raise ValueError("Expected an AMI .zip archive destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    if temp.exists():
        raise FileExistsError("Remove an incomplete .part file after inspection")
    try:
        with urllib.request.urlopen(ANNOTATION_URL, timeout=90) as source:
            with temp.open("xb") as writer:
                total = 0
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ZIP_BYTES:
                        raise ValueError("AMI archive exceeded download limit")
                    writer.write(chunk)
        with ZipFile(temp) as archive:
            if not archive.infolist():
                raise ValueError("Empty AMI manual annotation archive")
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()


def write_reference(
    words: list[HumanWord], metadata: dict, *,
    reference: Path, report: Path,
) -> dict:
    if reference.exists() or report.exists() or reference == report:
        raise FileExistsError("Refusing to overwrite human reference evidence")
    if reference.suffix != ".txt" or report.suffix != ".json":
        raise ValueError("Reference must be .txt and metadata must be .json")
    # The text contains all annotated lexical turns in start-time order,
    # including the overlapping phrases. Do not call it verified audible text.
    content = " ".join(word.text for word in words).strip() + "\n"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(content, encoding="utf-8", newline="\n")
    info = {
        "schema": SCHEMA,
        "human_reference_sha256": checksum(reference),
        "human_reference": str(reference),
        "verified_against_mixed_audio": False,
        **metadata,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("benchmark/fixed195/ami_shadow10"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download-if-needed", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    safe_root = Path("benchmark/fixed195").resolve()
    root = args.root.resolve()
    archive = (args.archive or (
        root / "ami_public_manual_1.6.2.zip"
    )).resolve()
    reference = (args.reference_output or (
        root / "TS3003b_1500_1740_official_orthographic.txt"
    )).resolve()
    metadata = (args.metadata_output or (
        root / "TS3003b_1500_1740_official_orthographic.json"
    )).resolve()
    if any(not path.is_relative_to(safe_root)
           for path in (root, archive, reference, metadata)):
        parser.error("All AMI files must remain under private benchmark/fixed195")
    if not archive.is_file():
        if not args.download_if_needed:
            parser.error("Download AMI manual v1.6.2 or use --download-if-needed")
        download_official_archive(archive)
    words, details = words_from_official_archive(archive)
    result = write_reference(words, details, reference=reference, report=metadata)
    print(json.dumps(result, indent=2))
    print(
        "IMPORTANT: reference is NOT yet reviewed against the mixed headset "
        "audio. Do not set -HumanReferenceVerified until it has been checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

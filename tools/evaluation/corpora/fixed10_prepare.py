"""Prepare ten immutable, licensed multilingual speech regression cases.

Sources (CC-BY-4.0):
- PolyAI/minds14 (eight short single-speaker utterances, eight locales)
- laion/small-overlapping-speech-bench (two three-speaker multilingual overlaps)

This is a fast smoke set, NOT sufficient for a production/all-language claim.
Source revisions and resulting WAV SHA-256 digests are saved in lock.json.
Re-running verifies existing bytes; it never silently replaces a frozen case.
Requires: pip install huggingface_hub pyarrow soundfile scipy numpy
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import wave

MIND_REPO = "PolyAI/minds14"
MIND_REV = "40ce77c"
OVERLAP_REPO = "laion/small-overlapping-speech-bench"
OVERLAP_REV = "250186c"
# Explicit, stable sample indices, independent of the machine and filesystem.
LOCALES = (
    ("en-GB", "en", 19), ("zh-CN", "zh", 19),
    ("es-ES", "es", 19), ("fr-FR", "fr", 19),
    ("de-DE", "de", 19), ("ko-KR", "ko", 19),
    ("pt-PT", "pt", 19), ("ru-RU", "ru", 19),
)
OVERLAPS = (0, 1)
SAMPLE_RATE = 16000


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav(path: Path, samples, sample_rate: int) -> float:
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Source contains invalid audio samples")
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(sample_rate, SAMPLE_RATE)
        a = resample_poly(a, SAMPLE_RATE // divisor, sample_rate // divisor)
    a = np.clip(a, -1.0, 1.0)
    packed = (a * 32767).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(packed)
    return round(len(a) / SAMPLE_RATE, 3)


def _download_file(repo: str, rev: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=repo, repo_type="dataset",
        revision=rev, filename=filename,
    ))


def _read_source_audio(raw: bytes):
    import soundfile as sf
    data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return data, int(rate)


def _manifest_write(target: Path, entries: list[dict]) -> None:
    target.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in entries),
        encoding="utf-8",
    )


def build(root: Path, *, verify_only: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    lock_file = root / "lock.json"
    singles_file = root / "manifest_single.jsonl"
    all_file = root / "cases.json"
    if lock_file.exists():
        saved = json.loads(lock_file.read_text(encoding="utf-8"))
        cases = saved["cases"]
        if len(cases) != 10:
            raise ValueError("The existing lock does not contain ten cases")
        for case in cases:
            path = root / case["audio"]
            if not path.is_file() or _checksum(path) != case["sha256"]:
                raise ValueError(f"Frozen audio missing or modified: {path}")
        if not all_file.is_file() or not singles_file.is_file():
            raise ValueError("Manifest missing; do not silently rebuild the frozen dataset")
        return saved
    if verify_only:
        raise FileNotFoundError("No frozen dataset exists yet")
    import pyarrow.parquet as pq
    cases: list[dict] = []
    singles: list[dict] = []
    for number, (locale, code, index) in enumerate(LOCALES, start=1):
        source = f"{locale}/train-00000-of-00001.parquet"
        dataset_file = _download_file(MIND_REPO, MIND_REV, source)
        tab = pq.read_table(dataset_file, columns=["audio", "transcription"])
        if index >= len(tab):
            raise ValueError(f"Row {index} is unavailable for {locale}")
        sample = tab.slice(index, 1).to_pylist()[0]
        audio = sample["audio"]
        if not isinstance(audio, dict) or not audio.get("bytes"):
            raise ValueError(f"{locale} has no embedded audio")
        raw_text = str(sample["transcription"] or "").strip()
        if not raw_text:
            raise ValueError(f"{locale} has an empty transcript")
        samples, sr = _read_source_audio(audio["bytes"])
        rel = f"audio/{number:02d}_{code}.wav"
        dst = root / rel
        duration = _wav(dst, samples, sr)
        if duration < 0.5:
            raise ValueError("Unexpectedly short audio")
        case = {
            "id": f"fixed10-{number:02d}", "kind": "single", "audio": rel,
            "language": code, "locale": locale, "reference": raw_text,
            "language_mode": "auto", "duration_s": duration,
            "sha256": _checksum(dst),
            "source_repo": MIND_REPO, "source_rev": MIND_REV,
            "source_file": source, "source_row": index,
        }
        singles.append({k: case[k] for k in
                        ("id", "audio", "reference", "language", "language_mode")})
        cases.append(case)
    ground_truth_file = _download_file(OVERLAP_REPO, OVERLAP_REV, "ground_truth.jsonl")
    ground_truth = [json.loads(x) for x in ground_truth_file.read_text(encoding="utf-8").splitlines() if x]
    for number, index in enumerate(OVERLAPS, start=9):
        item = next((x for x in ground_truth if x.get("clip_id") == f"clip_{index:03d}"), None)
        if item is None:
            raise ValueError(f"Missing source overlap clip {index}")
        filename = f"data/clip_{index:03d}.mp3"
        source = _download_file(OVERLAP_REPO, OVERLAP_REV, filename)
        # libsndfile can decode MP3 via soundfile on supported binary wheels.
        import soundfile as sf
        audio, sr = sf.read(str(source), dtype="float32", always_2d=False)
        rel = f"audio/{number:02d}_overlap.wav"
        duration = _wav(root / rel, audio, int(sr))
        speakers = []
        for speaker in item["speakers"]:
            speakers.append({
                "speaker": str(speaker["id"]),
                "language": str(speaker["code"]),
                "start_ms": round(float(speaker["start_time"]) * 1000),
                "end_ms": round(float(speaker["end_time"]) * 1000),
                "reference": str(speaker["transcript"]),
            })
        cases.append({
            "id": f"fixed10-{number:02d}", "kind": "overlap", "audio": rel,
            "language": "mul", "duration_s": duration,
            "sha256": _checksum(root / rel), "speakers": speakers,
            "source_repo": OVERLAP_REPO, "source_rev": OVERLAP_REV,
            "source_file": filename,
        })
    assert len(cases) == 10
    manifest = {"schema": "meeting-fixed10-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sample_rate": SAMPLE_RATE, "cases": cases,
                "license": "CC-BY-4.0",
                "attribution": [
                    "PolyAI MInDS-14: https://huggingface.co/datasets/PolyAI/minds14",
                    "LAION overlapping benchmark: https://huggingface.co/datasets/laion/small-overlapping-speech-bench",
                    "Underlying LAION speech derives from Multilingual LibriSpeech (Pratap et al., Interspeech 2020)",
                ]}
    # Manifest is written only after all ten audio files have been checked.
    _manifest_write(singles_file, singles)
    all_file.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed10"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = build(args.root, verify_only=args.verify_only)
    print(f"Verified {len(lock['cases'])} fixed WAVs in {args.root}")
    for case in lock["cases"]:
        print(f"{case['id']} {case['language']:>3} {case['duration_s']:6.2f}s  {case['sha256'][:16]}  {case['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

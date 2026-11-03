"""Prepare a reproducible 100-single + 50-multispeaker speech evaluation set.

Audio remains local and is never committed. The source revision, row, WAV SHA256,
acoustic condition and speaker transcripts are frozen in a lock file.
Downloads resume case-by-case and fail closed on changed files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import sys
import wave

MIND = "PolyAI/minds14"
FLEURS = "google/fleurs"
AMI = "Trelis/ami-2speaker-test"
LAION = "laion/small-overlapping-speech-bench"
MIND_LOCALES = (
    ("cs-CZ", "cs"), ("de-DE", "de"), ("en-AU", "en"),
    ("en-GB", "en"), ("en-US", "en"), ("es-ES", "es"),
    ("fr-FR", "fr"), ("it-IT", "it"), ("ko-KR", "ko"),
    ("nl-NL", "nl"), ("pl-PL", "pl"), ("pt-PT", "pt"),
    ("ru-RU", "ru"), ("zh-CN", "zh"),
)
# Additional writing systems / language families absent from MInDS-14.
FLEURS_LOCALES = (
    ("ar_eg", "ar"), ("hi_in", "hi"), ("ja_jp", "ja"),
    ("th_th", "th"), ("vi_vn", "vi"), ("tr_tr", "tr"),
    ("id_id", "id"), ("uk_ua", "uk"), ("bn_in", "bn"),
    ("fa_ir", "fa"), ("sw_ke", "sw"),
)
MIND_ROWS = (19, 43, 77, 101)
FLEURS_ROWS = (19, 43, 77, 101)
AMI_ROWS = tuple(range(0, 50, 2))
LAION_ROWS = tuple(range(0, 100, 4))
SAMPLE_RATE = 16000
TIMED = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)


def checksum(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_embedded_audio(value: dict):
    import soundfile as sf
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("Dataset row does not embed audio bytes; refusing an unverified external path")
    samples, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
    return samples, int(rate)


def write_wav(path: Path, samples, rate: int) -> float:
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Audio is empty, non-finite or not mono/stereo")
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, rate // g)
    data = np.clip(data, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((data * 32767).astype("<i2").tobytes())
    return round(len(data) / SAMPLE_RATE, 3)


def selected_parquet_rows(source: Path, indices: tuple[int, ...], columns: list[str]) -> dict[int, dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(source)
    if len(indices) != len(set(indices)):
        raise ValueError("Source row indices must be distinct")
    wanted = set(indices)
    found = {}
    offset = 0
    # Avoid decoding the entire 200+ MB FLEURS shard into memory.
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        matches = [i for i in wanted if offset <= i < offset + batch.num_rows]
        if matches:
            for row in matches:
                found[row] = batch.slice(row - offset, 1).to_pylist()[0]
        offset += batch.num_rows
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise ValueError(f"Missing rows {sorted(wanted - found.keys())} from {source.name}")
    return found


def ami_spans(target: str, speaker: str, duration: float) -> list[dict]:
    spans = []
    for match in TIMED.finditer(target):
        start, end = float(match.group(1)), float(match.group(3))
        sentence = match.group(2).strip()
        if sentence and 0 <= start < end <= duration + 0.35:
            spans.append({"start_ms": round(start * 1000),
                          "end_ms": round(end * 1000),
                          "speaker": speaker, "text": sentence})
    return spans


def verify_case(root: Path, case: dict) -> None:
    file = root / case["audio"]
    if not file.is_file() or checksum(file) != case["sha256"]:
        raise ValueError(f"Missing or modified frozen WAV: {case['id']}")
    with wave.open(str(file), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Wrong WAV format: {case['id']}")
        if f.getnframes() < 8000:
            raise ValueError(f"Unexpectedly short WAV: {case['id']}")


def verify_lock(root: Path) -> dict:
    lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
    cases = lock.get("cases", [])
    if (len(cases) != 150 or
        sum(c.get("kind") == "single" for c in cases) != 100 or
        sum(c.get("kind") == "multi" for c in cases) != 50 or
        len({c["id"] for c in cases}) != 150):
        raise ValueError("Fixed150 lock has an invalid 100+50 distribution")
    for case in cases:
        verify_case(root, case)
        if case["kind"] == "multi":
            if not len(case.get("speakers", [])) in (2, 3):
                raise ValueError(f"Bad speaker ground truth: {case['id']}")
    for filename in ("manifest_single.jsonl", "manifest_multi.jsonl"):
        if not (root / filename).is_file():
            raise ValueError(f"Missing manifest {filename}")
    return lock


def prepare(root: Path, *, verify_only: bool = False) -> dict:
    if (root / "lock.json").exists():
        return verify_lock(root)
    if verify_only:
        raise FileNotFoundError("Fixed150 has not finished preparation; see progress.json")
    # The caller controls HF_HOME / UV_CACHE_DIR / TEMP to avoid C: exhaustion.
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    root.mkdir(parents=True, exist_ok=True)
    progress_file = root / "progress.json"
    state = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {
        "schema": "meeting-fixed150-progress-v1",
        "revisions": {},
        "cases": {},
    }
    revisions = state["revisions"]
    for repo in (MIND, FLEURS, AMI, LAION):
        if repo not in revisions:
            revisions[repo] = api.dataset_info(repo).sha
            atomic_json(progress_file, state)

    def download(repo: str, filename: str) -> Path:
        return Path(hf_hub_download(repo_id=repo, repo_type="dataset",
                                    revision=revisions[repo], filename=filename))

    def commit(case: dict) -> None:
        verify_case(root, case)
        existing = state["cases"].get(case["id"])
        if existing and existing != case:
            raise ValueError(f"Source or metadata unexpectedly changed: {case['id']}")
        state["cases"][case["id"]] = case
        atomic_json(progress_file, state)
        print(f"{len(state['cases']):3}/150  {case['id']}  {case['source_repo']}  {case['duration_s']:.1f}s", flush=True)

    def is_done(cid: str) -> bool:
        if cid not in state["cases"]:
            return False
        verify_case(root, state["cases"][cid])
        return True

    number = 0
    for locale, language in MIND_LOCALES:
        group_ids = [f"fixed150-{x:03d}" for x in range(number + 1, number + 5)]
        if all(is_done(x) for x in group_ids):
            number += 4
            continue
        source_file = f"{locale}/train-00000-of-00001.parquet"
        rows = selected_parquet_rows(download(MIND, source_file), MIND_ROWS, ["audio", "transcription"])
        for row_id in MIND_ROWS:
            number += 1
            cid = f"fixed150-{number:03d}"
            if is_done(cid):
                continue
            data = rows[row_id]
            samples, rate = read_embedded_audio(data["audio"])
            rel = f"audio/{cid}.wav"
            duration = write_wav(root / rel, samples, rate)
            case = {
                "id": cid, "kind": "single", "audio": rel, "language": language,
                "locale": locale, "language_mode": "auto", "duration_s": duration,
                "reference": str(data["transcription"]).strip(),
                "scenario": "crowd_recorded_intent", "acoustic_condition": "original",
                "source_repo": MIND, "source_rev": revisions[MIND],
                "source_file": source_file, "source_row": row_id,
                "sha256": checksum(root / rel),
            }
            if not case["reference"]:
                raise ValueError(f"No reference for {cid}")
            commit(case)

    for locale, language in FLEURS_LOCALES:
        group_ids = [f"fixed150-{x:03d}" for x in range(number + 1, number + 5)]
        if all(is_done(x) for x in group_ids):
            number += 4
            continue
        source_file = f"{locale}/validation-00000-of-00001.parquet"
        rows = selected_parquet_rows(download(FLEURS, source_file), FLEURS_ROWS,
                                     ["audio", "transcription"])
        for row_id in FLEURS_ROWS:
            number += 1
            cid = f"fixed150-{number:03d}"
            if is_done(cid):
                continue
            data = rows[row_id]
            samples, rate = read_embedded_audio(data["audio"])
            rel = f"audio/{cid}.wav"
            duration = write_wav(root / rel, samples, rate)
            case = {
                "id": cid, "kind": "single", "audio": rel, "language": language,
                "locale": locale, "language_mode": "auto", "duration_s": duration,
                "reference": str(data["transcription"]).strip(),
                "scenario": "read_speech", "acoustic_condition": "original",
                "source_repo": FLEURS, "source_rev": revisions[FLEURS],
                "source_file": source_file, "source_row": row_id,
                "sha256": checksum(root / rel),
            }
            if not case["reference"]:
                raise ValueError(f"No reference for {cid}")
            commit(case)

    if number != 100:
        raise AssertionError("Single-speech allocation must be exactly 100")

    if not all(is_done(f"fixed150-{x:03d}") for x in range(101, 126)):
        source_file = "data/train-00000-of-00001.parquet"
        rows = selected_parquet_rows(download(AMI, source_file), AMI_ROWS,
                                     ["audio", "speaker1_text", "speaker2_text",
                                      "speaker1_target", "speaker2_target",
                                      "speaker1_start", "speaker1_end",
                                      "speaker2_start", "speaker2_end",
                                      "overlap_ratio"])
        for index, row_id in enumerate(AMI_ROWS, 101):
            cid = f"fixed150-{index:03d}"
            if is_done(cid):
                continue
            item = rows[row_id]
            samples, rate = read_embedded_audio(item["audio"])
            rel = f"audio/{cid}.wav"
            duration = write_wav(root / rel, samples, rate)
            speakers = []
            spans = []
            for j in (1, 2):
                identity = f"speaker_{j}"
                reference = str(item[f"speaker{j}_text"] or "").strip()
                if not reference:
                    raise ValueError(f"No speaker {j} transcript in {cid}")
                stamps = ami_spans(str(item[f"speaker{j}_target"] or ""), identity, duration)
                if not stamps:
                    raise ValueError(f"Missing time-coded AMI segments in {cid}")
                spans.extend(stamps)
                speakers.append({
                    "speaker": identity, "language": "en", "reference": reference,
                    "start_ms": min(s["start_ms"] for s in stamps),
                    "end_ms": max(s["end_ms"] for s in stamps),
                })
            commit({
                "id": cid, "kind": "multi", "audio": rel, "language": "en",
                "duration_s": duration, "scenario": "real_meeting_two_speakers",
                "acoustic_condition": "original", "speaker_count": 2,
                "speakers": speakers,
                "reference_segments": sorted(spans, key=lambda s: (s["start_ms"], s["speaker"])),
                "overlap_ratio": float(item["overlap_ratio"]),
                "annotation_precision": "utterance",
                "source_repo": AMI, "source_rev": revisions[AMI],
                "source_file": source_file, "source_row": row_id,
                "sha256": checksum(root / rel),
            })

    if not all(is_done(f"fixed150-{x:03d}") for x in range(126, 151)):
        ground = [json.loads(x) for x in download(LAION, "ground_truth.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]
        by_id = {row["clip_id"]: row for row in ground}
        for index, source_row in enumerate(LAION_ROWS, 126):
            cid = f"fixed150-{index:03d}"
            if is_done(cid):
                continue
            clip_id = f"clip_{source_row:03d}"
            if clip_id not in by_id:
                raise ValueError(f"Missing LAION ground truth: {clip_id}")
            item = by_id[clip_id]
            source_file = f"data/{clip_id}.mp3"
            import soundfile as sf
            samples, rate = sf.read(str(download(LAION, source_file)), dtype="float32")
            rel = f"audio/{cid}.wav"
            duration = write_wav(root / rel, samples, int(rate))
            speakers = []
            for sp in item["speakers"]:
                speakers.append({
                    "speaker": str(sp["id"]), "language": str(sp["code"]),
                    "start_ms": round(float(sp["start_time"]) * 1000),
                    "end_ms": round(float(sp["end_time"]) * 1000),
                    "reference": str(sp["transcript"]),
                })
            if len(speakers) != 3:
                raise ValueError(f"LAION clip lacks three speakers: {clip_id}")
            commit({
                "id": cid, "kind": "multi", "audio": rel, "language": "mul",
                "duration_s": duration, "scenario": "synthetic_three_language_overlap",
                "acoustic_condition": "original", "speaker_count": 3,
                "speakers": speakers,
                "reference_segments": [
                    {"speaker": s["speaker"], "start_ms": s["start_ms"],
                     "end_ms": s["end_ms"], "text": s["reference"]}
                    for s in speakers
                ],
                "annotation_precision": "source_utterance",
                "source_repo": LAION, "source_rev": revisions[LAION],
                "source_file": source_file, "source_row": source_row,
                "sha256": checksum(root / rel),
            })

    cases = [state["cases"][f"fixed150-{i:03d}"] for i in range(1, 151)]
    if len({x["sha256"] for x in cases}) != 150:
        raise ValueError("Duplicate audio content found; refusing to freeze")
    for case in cases:
        verify_case(root, case)
    single_manifest = [
        {k: c[k] for k in ("id", "audio", "reference", "language", "language_mode",
                            "scenario", "locale")}
        for c in cases if c["kind"] == "single"
    ]
    multi_manifest = [
        {k: c[k] for k in ("id", "audio", "language", "speaker_count", "scenario")}
        for c in cases if c["kind"] == "multi"
    ]
    for name, rows in (("manifest_single.jsonl", single_manifest),
                       ("manifest_multi.jsonl", multi_manifest)):
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        (root / name).write_text(text, encoding="utf-8")
    atomic_json(root / "lock.json", {
        "schema": "meeting-fixed150-v1", "sample_rate": 16000,
        "source_revisions": revisions, "cases": cases,
        "license": "CC-BY-4.0",
        "attribution": [
            "PolyAI MInDS-14", "Google FLEURS", "AMI Meeting Corpus (Carletta et al.)",
            "LAION Small Overlapping Speech Bench (source MLS, Pratap et al.)",
        ],
        "limitations": [
            "Not a representative statistical sample of all languages or microphones",
            "AMI has two speakers with as-recorded meeting audio",
            "LAION overlapping sources are synthetically combined read speech",
            "No source proves a specific room or noise condition without annotation",
        ],
    })
    return verify_lock(root)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if os.name == "nt":
        print("HF_HOME:", os.environ.get("HF_HOME", "(default: may use C drive)"))
        print("TEMP:", os.environ.get("TEMP", "(default)"))
    lock = prepare(args.root, verify_only=args.verify_only)
    counts = Counter(c["scenario"] for c in lock["cases"])
    print("LOCK VERIFIED: 150 unique WAV files, 100 singles, 50 multispeakers")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

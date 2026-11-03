"""Prepare 12-language Fixed120 benchmark with 60 single and 60 real dialogs.

Source A: Google FLEURS validation, five single-speaker clips per language.
Source B: David AI DAI-ASR-I18N, five genuine two-speaker clips per language.

DAI access is gated and subject to its Data Use Agreement: evaluation only,
no model training/optimization, no redistribution or identification. Audio,
references and all generated files stay local; only this source code is
suitable for GitHub. Never turn these dialogues into synthetic conversations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from tools.fixed150_prepare import (
    atomic_json, checksum, choose_source_file, read_embedded_audio,
    selected_parquet_rows, write_wav,
)

FLEURS = "google/fleurs"
DAI = "davidai-labs/DAI-ASR-I18N"
LANGUAGES = (
    ("en", "en_us"), ("zh", "cmn_hans_cn"), ("hi", "hi_in"),
    ("es", "es_419"), ("fr", "fr_fr"), ("ar", "ar_eg"),
    ("bn", "bn_in"), ("pt", "pt_br"), ("ru", "ru_ru"),
    ("id", "id_id"), ("de", "de_de"), ("ja", "ja_jp"),
)
# Deliberately disjoint from the original Fixed150 FLEURS rows 19/43/77/101.
FLEURS_VALIDATION_ROWS = (113, 121, 129, 137, 145)
MULTI_QUANTILES = (.10, .30, .50, .70, .90)
SCHEMA = "meeting-fixed120-real-v1"


def choose_dialogues(rows: list[dict], lang: str) -> list[dict]:
    """Select five genuine dialogues by overlap diversity, not ASR scores."""
    accepted = []
    seen = set()
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if not sid or sid in seen or str(row.get("language", "")).lower() != lang:
            continue
        seen.add(sid)
        if row.get("n_speakers") != 2:
            continue
        if not row.get("speaker1_id") or not row.get("speaker2_id"):
            continue
        if row["speaker1_id"] == row["speaker2_id"]:
            continue
        duration = row.get("duration_s")
        overlap = row.get("clip_overlap_frac")
        try:
            duration, overlap = float(duration), float(overlap)
        except (ValueError, TypeError, OverflowError):
            continue
        if (not math.isfinite(duration) or not math.isfinite(overlap)
                or duration < 12 or overlap < 0 or overlap > 1):
            continue
        segments = collect_segments(row.get("word_alignments"))
        if {x["speaker"] for x in segments} != {"speaker_1", "speaker_2"}:
            continue
        # Require real speech from both participants, not a silent channel.
        if any(sum(len(x["words"]) for x in segments
                   if x["speaker"] == speaker) < 4
               for speaker in ("speaker_1", "speaker_2")):
            continue
        accepted.append((overlap, sid, row))
    if len(accepted) < 5:
        raise ValueError(
            f"Only {len(accepted)} valid two-speaker conversations for {lang}; "
            "do not replace missing real dialogs with generated mixtures."
        )
    accepted.sort(key=lambda x: (x[0], x[1]))
    indices = [round((len(accepted)-1) * q) for q in MULTI_QUANTILES]
    if len(set(indices)) != 5:
        raise ValueError("Cannot select five distinct overlap quantiles")
    return [accepted[i][2] for i in indices]


def collect_segments(value: object) -> list[dict]:
    """Convert human-verified word alignments into canonical speaker spans."""
    if not isinstance(value, list):
        return []
    results = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            speaker = int(item.get("speaker"))
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (ValueError, TypeError, OverflowError):
            continue
        if speaker not in (1, 2) or not (math.isfinite(start)
                                         and math.isfinite(end)
                                         and 0 <= start < end):
            continue
        words = []
        for w in item.get("words") or []:
            if not isinstance(w, dict):
                continue
            token = str(w.get("text") or "").strip()
            try:
                wstart, wend = float(w.get("start")), float(w.get("end"))
            except (ValueError, TypeError, OverflowError):
                continue
            if (not token or not math.isfinite(wstart)
                    or not math.isfinite(wend) or wstart >= wend
                    or wstart < 0):
                continue
            words.append({
                "start_ms": round(wstart * 1000),
                "end_ms": round(wend * 1000),
                "text": token,
            })
        if words:
            results.append({
                "speaker": f"speaker_{speaker}",
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "timing_source": item.get("timing_source"),
                "text": " ".join(w["text"] for w in words),
                "words": words,
            })
    results.sort(key=lambda x: (x["start_ms"], x["speaker"]))
    return results


def mix_tracks(left: Path, right: Path, target: Path) -> tuple[float, float]:
    """Locally mix synchronized real speakers, preserving their full timeline."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    streams = []
    for source in (left, right):
        audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError("DAI channel expected to be mono")
        mono = audio[:, 0]
        if rate != 16000:
            g = gcd(rate, 16000)
            mono = resample_poly(mono, 16000 // g, rate // g)
        if len(mono) < 16000 or not np.all(np.isfinite(mono)):
            raise ValueError("Invalid speaker channel")
        streams.append(mono)
    length = max(map(len, streams))
    if abs(len(streams[0]) - len(streams[1])) > 1600:
        raise ValueError("Speaker channel clocks differ by more than 100 ms")
    mixed = np.zeros(length, dtype=np.float32)
    for track in streams:
        mixed[:len(track)] += track
    peak = float(np.max(np.abs(mixed))) if length else 0.0
    gain = min(1.0, .98 / peak) if peak else 1.0
    duration = write_wav(target, mixed * gain, 16000)
    return duration, round(gain, 6)


def manifest_entry(root: Path, language: str, index: int, mode: str) -> dict:
    cid = f"fixed120-{language}-{mode}{index}"
    return {"id": cid, "language": language, "kind": "single" if mode == "s" else "multi"}


def verify_case(root: Path, item: dict) -> None:
    import wave

    path = root / item["audio"]
    if not path.is_file() or checksum(path) != item["sha256"]:
        raise ValueError(f"Frozen audio missing/changed: {item['id']}")
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(),
                stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Unexpected WAV geometry: {item['id']}")
    if item["kind"] == "multi":
        speakers = {x["speaker"] for x in item["reference_segments"]}
        if speakers != {"speaker_1", "speaker_2"}:
            raise ValueError(f"Missing real speaker reference: {item['id']}")


def create_outputs(root: Path, state: dict) -> bool:
    singles, multi = [], []
    for language, _ in LANGUAGES:
        for idx in range(1, 6):
            for mode in ("s", "m"):
                cid = f"fixed120-{language}-{mode}{idx}"
                case = state["cases"].get(cid)
                if not case:
                    continue
                verify_case(root, case)
                (singles if mode == "s" else multi).append(case)
    for name, rows in (("manifest_single.jsonl", singles),
                       ("manifest_multi.jsonl", multi)):
        target = root / name
        target.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
            encoding="utf-8",
        )
    complete = len(singles) == 60 and len(multi) == 60
    if complete:
        atomic_json(root / "lock.json", {
            "schema": SCHEMA, "revisions": state["revisions"],
            "cases": [x for lang, _ in LANGUAGES for x in singles + multi
                      if x["language"] == lang],
            "evaluation_only": True,
        })
    print(f"Fixed120 available: singles={len(singles)}/60 "
          f"real_dialogues={len(multi)}/60  complete={complete}")
    return complete


def prepare(root: Path, source: str, verify_only: bool) -> bool:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    progress = root / "progress.json"
    state = (json.loads(progress.read_text(encoding="utf-8"))
             if progress.exists()
             else {"schema": SCHEMA, "revisions": {}, "cases": {}})
    if state.get("schema") != SCHEMA:
        raise ValueError("Incompatible Fixed120 checkpoint")
    if verify_only:
        if not (root / "lock.json").exists():
            raise FileNotFoundError("Fixed120 lock not finished")
        lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
        if lock.get("schema") != SCHEMA or len(lock.get("cases", [])) != 120:
            raise ValueError("Incorrect Fixed120 lock")
        for item in lock["cases"]:
            verify_case(root, item)
        print("Fixed120 verified: 120 unique clips, valid SHA-256 and speaker references")
        return True

    api = HfApi()
    requested = ([FLEURS] if source == "singles" else
                 [DAI] if source == "multi" else [FLEURS, DAI])
    for repository in requested:
        if repository not in state["revisions"]:
            try:
                state["revisions"][repository] = api.dataset_info(repository).sha
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot pin dataset {repository}. For DAI, request access "
                    "and authenticate with hf auth login before running."
                ) from exc
            atomic_json(progress, state)

    def download(repo: str, filename: str) -> Path:
        try:
            return Path(hf_hub_download(
                repo_id=repo, repo_type="dataset",
                revision=state["revisions"][repo], filename=filename,
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download {repo}/{filename}. For the gated DAI corpus, "
                "request access and accept the Data Use Agreement first."
            ) from exc

    def commit(item: dict) -> None:
        verify_case(root, item)
        old = state["cases"].get(item["id"])
        if old and old != item:
            raise ValueError(f"Source mismatch: {item['id']}")
        state["cases"][item["id"]] = item
        atomic_json(progress, state)
        print(f"  {item['id']} {item['duration_s']:.1f}s "
              f"{item['source_repo']}", flush=True)

    if FLEURS in requested:
        inventory = set(api.list_repo_files(
            repo_id=FLEURS, repo_type="dataset",
            revision=state["revisions"][FLEURS],
        ))
        for language, locale in LANGUAGES:
            ids = [f"fixed120-{language}-s{i}" for i in range(1, 6)]
            if all(cid in state["cases"] for cid in ids):
                for cid in ids:
                    verify_case(root, state["cases"][cid])
                continue
            filename = choose_source_file(
                inventory, f"{locale}/validation-00000-of-00001.parquet",
                repo=FLEURS, revision=state["revisions"][FLEURS],
            )
            entries = selected_parquet_rows(
                download(FLEURS, filename), FLEURS_VALIDATION_ROWS,
                ["id", "audio", "transcription"],
            )
            for index, row_id in enumerate(FLEURS_VALIDATION_ROWS, 1):
                cid = f"fixed120-{language}-s{index}"
                if cid in state["cases"]:
                    verify_case(root, state["cases"][cid])
                    continue
                data = entries[row_id]
                text = str(data.get("transcription") or "").strip()
                if not text:
                    raise ValueError(f"Empty single-speaker transcription: {cid}")
                samples, sr = read_embedded_audio(data["audio"])
                relative = f"audio/{cid}.wav"
                duration = write_wav(root / relative, samples, sr)
                commit({
                    "id": cid, "language": language, "locale": locale,
                    "kind": "single", "audio": relative,
                    "duration_s": duration, "reference": text,
                    "scenario": "fleurs_real_read_speech",
                    "source_repo": FLEURS, "source_rev": state["revisions"][FLEURS],
                    "source_file": filename, "source_row": row_id,
                    "source_sample_id": data["id"],
                    "sha256": checksum(root / relative),
                })

    if DAI in requested:
        for language, _ in LANGUAGES:
            ids = [f"fixed120-{language}-m{i}" for i in range(1, 6)]
            if all(cid in state["cases"] for cid in ids):
                for cid in ids:
                    verify_case(root, state["cases"][cid])
                continue
            filename = f"data/{language}/{language}.parquet"
            table = pq.read_table(download(DAI, filename))
            chosen = choose_dialogues(table.to_pylist(), language)
            for index, row in enumerate(chosen, 1):
                cid = f"fixed120-{language}-m{index}"
                if cid in state["cases"]:
                    verify_case(root, state["cases"][cid])
                    if state["cases"][cid]["source_sample_id"] != row["sample_id"]:
                        raise ValueError(f"Conversation selection changed: {cid}")
                    continue
                sample_id = row["sample_id"]
                ch1 = download(DAI, f"audio/{sample_id}/ch1.flac")
                ch2 = download(DAI, f"audio/{sample_id}/ch2.flac")
                relative = f"audio/{cid}.wav"
                duration, gain = mix_tracks(ch1, ch2, root / relative)
                reference_segments = collect_segments(row["word_alignments"])
                speaker_refs = [
                    {"speaker": speaker, "reference": " ".join(
                        seg["text"] for seg in reference_segments
                        if seg["speaker"] == speaker
                    )} for speaker in ("speaker_1", "speaker_2")
                ]
                commit({
                    "id": cid, "language": language,
                    "locale": row.get("locale"), "kind": "multi",
                    "audio": relative, "duration_s": duration,
                    "scenario": "real_two_speaker_conversation",
                    "speaker_count": 2,
                    "reference_segments": reference_segments,
                    "speakers": speaker_refs,
                    "overlap_type": row.get("overlap_type"),
                    "overlap_fraction": row.get("clip_overlap_frac"),
                    "mix_gain": gain,
                    "timing_contains_pseudo": any(
                        s["timing_source"] == "pseudo"
                        for s in reference_segments
                    ),
                    "source_repo": DAI, "source_rev": state["revisions"][DAI],
                    "source_sample_id": sample_id,
                    "source_file": filename,
                    "sha256": checksum(root / relative),
                    "license_restrictions": "evaluation only; no training, optimization or redistribution",
                })
    return create_outputs(root, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed120"))
    parser.add_argument("--source", choices=("singles", "multi", "all"), default="singles")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    complete = prepare(args.root, args.source, args.verify)
    return 0 if complete or args.source == "singles" else 2


if __name__ == "__main__":
    raise SystemExit(main())

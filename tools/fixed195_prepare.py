"""Prepare 130 verified original and new single-speaker test clips.

Fixed195's authentic two-speaker audio collection is delegated to the
separate ungated public conversation collector. Audio-only clips must never
be represented as verified speaker-attributed transcript references.
Old Fixed150 and legacy Fixed120 benchmark data remain immutable.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fixed150_prepare import (
    atomic_json, checksum, choose_source_file, read_embedded_audio,
    selected_parquet_rows, write_wav, verify_lock as verify_fixed150,
)

FLEURS = "google/fleurs"
LANGUAGES = (
    ("en", "en_us"), ("zh", "cmn_hans_cn"), ("hi", "hi_in"),
    ("es", "es_419"), ("fr", "fr_fr"), ("ar", "ar_eg"),
    ("bn", "bn_in"), ("pt", "pt_br"), ("ru", "ru_ru"),
    ("id", "id_id"), ("de", "de_de"), ("ja", "ja_jp"),
)
# Six new examples per language, disjoint from Fixed150 FLEURS validation rows.
# The first five also match the previous, unfinished Fixed120 preparer.
FLEURS_VALIDATION_ROWS = (113, 121, 129, 137, 145, 153)
ENGLISH_FLEURS_ROWS = FLEURS_VALIDATION_ROWS + (161, 169)
SCHEMA = "meeting-fixed195-real-v1"

# Use all twelve previously tested English originals from Australia, the
# United Kingdom and the United States. Other languages retain four each.
REUSED_FIXED150_IDS = {
    "en": tuple(f"fixed150-{i:03d}" for i in range(9, 21)),
    "zh": ("fixed150-053", "fixed150-054", "fixed150-055", "fixed150-056"),
    "hi": ("fixed150-061", "fixed150-062", "fixed150-063", "fixed150-064"),
    "es": ("fixed150-021", "fixed150-022", "fixed150-023", "fixed150-024"),
    "fr": ("fixed150-025", "fixed150-026", "fixed150-027", "fixed150-028"),
    "ar": ("fixed150-057", "fixed150-058", "fixed150-059", "fixed150-060"),
    "bn": ("fixed150-089", "fixed150-090", "fixed150-091", "fixed150-092"),
    "pt": ("fixed150-045", "fixed150-046", "fixed150-047", "fixed150-048"),
    "ru": ("fixed150-049", "fixed150-050", "fixed150-051", "fixed150-052"),
    "id": ("fixed150-081", "fixed150-082", "fixed150-083", "fixed150-084"),
    "de": ("fixed150-005", "fixed150-006", "fixed150-007", "fixed150-008"),
    "ja": ("fixed150-065", "fixed150-066", "fixed150-067", "fixed150-068"),
}
SINGLES_PER_LANGUAGE = 10
MULTI_PER_LANGUAGE = 5
ENGLISH_SINGLES = 20
ENGLISH_MULTI = 10


def singles_for(language: str) -> int:
    return ENGLISH_SINGLES if language == "en" else SINGLES_PER_LANGUAGE


def multi_for(language: str) -> int:
    return ENGLISH_MULTI if language == "en" else MULTI_PER_LANGUAGE


def rows_for(language: str) -> tuple[int, ...]:
    return ENGLISH_FLEURS_ROWS if language == "en" else FLEURS_VALIDATION_ROWS


def quantiles_for(language: str) -> tuple[float, ...]:
    n = multi_for(language)
    return tuple((i + 0.5) / n for i in range(n))


def choose_dialogues(rows: list[dict], lang: str) -> list[dict]:
    """Select five (or ten English) authentic dialogues by overlap diversity."""
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
    required = multi_for(lang)
    if len(accepted) < required:
        raise ValueError(
            f"Only {len(accepted)} valid two-speaker conversations for {lang}; "
            "do not replace missing real dialogs with generated mixtures."
        )
    accepted.sort(key=lambda x: (x[0], x[1]))
    indices = [round((len(accepted)-1) * q) for q in quantiles_for(lang)]
    if len(set(indices)) != required:
        raise ValueError("Cannot select distinct overlap quantiles")
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
        for idx in range(1, singles_for(language) + 1):
            cid = f"fixed195-{language}-s{idx:02d}"
            case = state["cases"].get(cid)
            if case:
                verify_case(root, case)
                singles.append(case)
        for idx in range(1, multi_for(language) + 1):
            cid = f"fixed195-{language}-m{idx:02d}"
            case = state["cases"].get(cid)
            if case:
                verify_case(root, case)
                multi.append(case)
    for name, rows in (("manifest_single.jsonl", singles),
                       ("manifest_multi.jsonl", multi)):
        target = root / name
        target.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
            encoding="utf-8",
        )
    complete = len(singles) == 130 and len(multi) == 65
    if complete:
        all_cases = [x for lang, _ in LANGUAGES for x in singles + multi
                     if x["language"] == lang]
        if len({x["id"] for x in all_cases}) != 195:
            raise ValueError("Duplicate Fixed195 case IDs")
        if len({x["sha256"] for x in all_cases}) != 195:
            raise ValueError("Duplicate audio content in Fixed195")
        if any(sum(x["kind"] == "single" for x in all_cases
                   if x["language"] == lang) != singles_for(lang) or
               sum(x["kind"] == "multi" for x in all_cases
                   if x["language"] == lang) != multi_for(lang)
               for lang, _ in LANGUAGES):
            raise ValueError("Language allocation is not 20+10 English, 10+5 others")
        atomic_json(root / "lock.json", {
            "schema": SCHEMA, "revisions": state["revisions"],
            "cases": all_cases, "evaluation_only": True,
            "reused_fixed150_singles": 56,
            "new_fleurs_singles": 74,
            "real_two_speaker_conversations": 65,
        })
    print(f"Fixed195 available: singles={len(singles)}/130 "
          f"real_dialogues={len(multi)}/65  complete={complete}")
    return complete


def prepare(root: Path, source: str, verify_only: bool,
            old_root: Path = Path("benchmark/fixed150"),
            previous_root: Path = Path("benchmark/fixed120")) -> bool:
    from huggingface_hub import HfApi, hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    progress = root / "progress.json"
    state = (json.loads(progress.read_text(encoding="utf-8"))
             if progress.exists()
             else {"schema": SCHEMA, "revisions": {}, "cases": {}})
    if state.get("schema") != SCHEMA:
        raise ValueError("Incompatible Fixed195 checkpoint")
    if verify_only:
        if not (root / "lock.json").exists():
            raise FileNotFoundError("Fixed195 lock not finished")
        lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
        cases = lock.get("cases", [])
        if lock.get("schema") != SCHEMA or len(cases) != 195:
            raise ValueError("Incorrect Fixed195 lock")
        if len({c["id"] for c in cases}) != 195 or len({c["sha256"] for c in cases}) != 195:
            raise ValueError("Fixed195 duplicate IDs or audio hashes")
        for item in cases:
            verify_case(root, item)
        for language, _ in LANGUAGES:
            matching = [c for c in cases if c["language"] == language]
            if (len(matching) != singles_for(language) + multi_for(language) or
                sum(c["kind"] == "single" for c in matching) != singles_for(language) or
                sum(c["kind"] == "multi" for c in matching) != multi_for(language)):
                raise ValueError(f"Incorrect Fixed195 allocation for {language}")
            original = [c for c in matching if c.get("reused_fixed150_id")]
            if [c["reused_fixed150_id"] for c in original] != list(REUSED_FIXED150_IDS[language]):
                raise ValueError(f"Original Fixed150 provenance mismatch for {language}")
        print("Fixed195 verified: 130 single + 65 genuine dual-speaker recordings")
        return True

    if source != "singles":
        raise ValueError(
            "Real public conversations are audio-only; use "
            "tools/fixed195_public_multi.py for non-scored source material."
        )
    api = HfApi()
    requested = [FLEURS]
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
        # Import all 56 eligible originals, including 12 English accents,
        # without regenerating source rows or counting any clip twice.
        old_lock = verify_fixed150(old_root)
        old_singles = {x["id"]: x for x in old_lock["cases"]
                       if x["kind"] == "single"}
        all_selected = [cid for ids in REUSED_FIXED150_IDS.values() for cid in ids]
        if len(all_selected) != 56 or len(set(all_selected)) != 56:
            raise ValueError("Fixed150 selection is not exactly 56 unique clips")
        for language, _ in LANGUAGES:
            for index, old_id in enumerate(REUSED_FIXED150_IDS[language], 1):
                original = old_singles.get(old_id)
                if not original or original["language"] != language:
                    raise ValueError(f"Original language/source mismatch for {old_id}")
                cid = f"fixed195-{language}-s{index:02d}"
                if cid in state["cases"]:
                    verify_case(root, state["cases"][cid])
                    if state["cases"][cid].get("reused_fixed150_id") != old_id:
                        raise ValueError(f"Original source changed: {cid}")
                    continue
                relative = f"audio/{cid}.wav"
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old_root / original["audio"], dest)
                if checksum(dest) != original["sha256"]:
                    raise ValueError(f"Old source differs from frozen SHA: {old_id}")
                item = {
                    **original,
                    "id": cid,
                    "audio": relative,
                    "reused_fixed150_id": old_id,
                    "reused_fixed150_sha256": original["sha256"],
                    "benchmark_role": "retained_original_single",
                }
                commit(item)

        previous_cases = {}
        previous_progress = previous_root / "progress.json"
        if previous_progress.exists():
            previous = json.loads(previous_progress.read_text(encoding="utf-8"))
            if previous.get("schema") == "meeting-fixed120-real-v1":
                previous_cases = previous.get("cases", {})

        inventory = set(api.list_repo_files(
            repo_id=FLEURS, repo_type="dataset",
            revision=state["revisions"][FLEURS],
        ))
        for language, locale in LANGUAGES:
            start_index = len(REUSED_FIXED150_IDS[language]) + 1
            ids = [
                f"fixed195-{language}-s{i:02d}"
                for i in range(start_index, singles_for(language) + 1)
            ]
            if all(cid in state["cases"] for cid in ids):
                for cid in ids:
                    verify_case(root, state["cases"][cid])
                continue
            filename = choose_source_file(
                inventory, f"{locale}/validation-00000-of-00001.parquet",
                repo=FLEURS, revision=state["revisions"][FLEURS],
            )
            entries = selected_parquet_rows(
                download(FLEURS, filename), rows_for(language),
                ["id", "audio", "transcription"],
            )
            for new_index, row_id in enumerate(rows_for(language), 1):
                index = len(REUSED_FIXED150_IDS[language]) + new_index
                cid = f"fixed195-{language}-s{index:02d}"
                if cid in state["cases"]:
                    verify_case(root, state["cases"][cid])
                    if state["cases"][cid].get("source_row") != row_id:
                        raise ValueError(f"New FLEURS selection changed: {cid}")
                    continue

                # Reuse pre-existing Fixed120 singles where they are
                # identical in pinned dataset revision, locale and row.
                old_cid = f"fixed120-{language}-s{new_index}"
                prior = previous_cases.get(old_cid)
                if (prior and prior.get("source_repo") == FLEURS
                    and prior.get("source_rev") == state["revisions"][FLEURS]
                    and prior.get("locale") == locale
                    and prior.get("source_row") == row_id):
                    from tools.fixed120_prepare import verify_case as verify_previous
                    verify_previous(previous_root, prior)
                    relative = f"audio/{cid}.wav"
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(previous_root / prior["audio"], target)
                    commit({
                        **prior, "id": cid, "audio": relative,
                        "reused_fixed120_id": old_cid,
                        "benchmark_role": "new_fleurs_single",
                    })
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
                    "benchmark_role": "new_fleurs_single",
                    "duration_s": duration, "reference": text,
                    "scenario": "fleurs_real_read_speech",
                    "source_repo": FLEURS, "source_rev": state["revisions"][FLEURS],
                    "source_file": filename, "source_row": row_id,
                    "source_sample_id": data["id"],
                    "sha256": checksum(root / relative),
                })

    return create_outputs(root, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--old-root", type=Path, default=Path("benchmark/fixed150"))
    parser.add_argument("--previous-root", type=Path, default=Path("benchmark/fixed120"))
    parser.add_argument("--source", choices=("singles", "multi", "all"), default="singles")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.source == "multi":
        # Public dual-channel audio is a staged collection, not a scored lock.
        # Never read gated DAI metadata or fabricate reference transcripts.
        from tools.fixed195_public_multi import prepare as prepare_public_multi
        prepare_public_multi(args.root)
        return 0
    if not args.verify and args.source == "all":
        # Preserve the 130 singles and gather all currently open authentic
        # conversations. Full Fixed195 still requires independently verified
        # speaker-attributed transcripts and the remaining languages.
        prepare(args.root, "singles", False, args.old_root, args.previous_root)
        from tools.fixed195_public_multi import prepare as prepare_public_multi
        prepare_public_multi(args.root)
        return 0
    complete = prepare(args.root, args.source, args.verify,
                       args.old_root, args.previous_root)
    return 0 if complete or args.source == "singles" else 2


if __name__ == "__main__":
    raise SystemExit(main())

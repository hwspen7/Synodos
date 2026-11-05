"""Freeze independent Google FLEURS multilingual ASR cases to the E drive.

Optional network work is explicit. No model inference, no train split, no
selection on predictions, no automatic quality/pass labels.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import wave

SCHEMA = "meeting-asr-selection-corpus-v1"
LANGUAGES = {
    "en": "en_us", "zh": "cmn_hans_cn", "fr": "fr_fr",
    "es": "es_419", "ar": "ar_eg", "id": "id_id",
    "ja": "ja_jp", "ru": "ru_ru", "de": "de_de", "hi": "hi_in",
}
ROOT = Path(r"E:\AI-Cache\MeetingReference\ASRSelection")
DATASET = "google/fleurs"
ALLOWED_SPLITS = {"validation": "development", "test": "sealed_test"}
SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
MIN_SECONDS, MAX_SECONDS = 0.8, 30.0
FREEZE_VERSION = "sf_v3"
PLAN_SCHEMA = "meeting-fleurs-freeze-plan-v3"


def require_e_drive(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and resolved.drive.casefold() != "e:":
        raise ValueError("ASR benchmark datasets must stay on the E drive")
    return resolved


def configure_cache(root: Path) -> None:
    cache = root / "cache"
    for key, value in (
        ("HF_HOME", cache / "huggingface"),
        ("HF_HUB_CACHE", cache / "huggingface" / "hub"),
        ("HF_DATASETS_CACHE", cache / "huggingface" / "datasets"),
        ("TORCH_HOME", cache / "torch"),
        ("XDG_CACHE_HOME", cache / "xdg"),
    ):
        existing = os.getenv(key, "")
        if existing and os.name == "nt" and Path(existing).resolve().drive.casefold() != "e:":
            raise ValueError(key + " points outside E drive")
        os.environ[key] = str(require_e_drive(Path(existing) if existing else value))


def choose_indices(metadata, *, language: str, count: int, seed: str) -> list[int]:
    """Preselect independent of transcripts, audio content and predictions."""
    ordered = sorted(
        ((hashlib.sha256(
            f"{seed}/{language}/{row['id']}".encode("utf-8")
        ).hexdigest(), i) for i, row in enumerate(metadata)),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [index for _, index in ordered[:count]]



def decode_audio_without_torchcodec(value: dict) -> dict:
    """Read FLEURS encoded audio with libsndfile, not the PyTorch ABI.

    This avoids TorchCodec, which has no Windows cp312 wheel compatible
    with the Meeting production torch 2.7 CUDA stack.
    """
    import numpy as np
    import soundfile as sf

    if not isinstance(value, dict):
        raise ValueError("Unexpected encoded audio record")
    data = value.get("bytes")
    if data is not None:
        if not isinstance(data, bytes):
            raise ValueError("Encoded audio bytes must be bytes")
        reader = io.BytesIO(data)
    else:
        filename = value.get("path")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Encoded audio has no bytes or source file")
        reader = filename
    try:
        samples, rate = sf.read(reader, dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as exc:
        raise ValueError(
            "Audio decode failed in libsndfile. Verify soundfile MP3 support; "
            "do not install datasets[audio] or torchcodec in Meeting .venv."
        ) from exc
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("FLEURS source audio must be mono")
    if not np.isfinite(samples).all() or samples.size == 0:
        raise ValueError("Empty or invalid source audio")
    if rate != 16000:
        # Preserve all selected examples instead of silently skipping
        # files based on their original codec's sampling frequency.
        from scipy.signal import resample_poly
        scale = math.gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // scale,
                                int(rate) // scale).astype(np.float32)
    return {"array": samples, "sampling_rate": 16000}




def choose_streaming_examples(dataset, *, language: str, count: int,
                              seed: str) -> list[dict]:
    """Select smallest SHA256 ranks without reference or audio scoring.

    Keep only the requested amount of un-decoded source records in memory.
    Ties are broken by source index, matching choose_indices().
    """
    if count < 1:
        raise ValueError("Selection must have positive size")
    selected = []
    for index, example in enumerate(dataset):
        if not isinstance(example, dict) or "id" not in example:
            raise ValueError("Streaming FLEURS row has no official ID")
        if type(example["id"]) is not int or example["id"] < 0:
            raise ValueError("Invalid official FLEURS sample ID")
        source_id = example["id"]
        # FLEURS' official ID field may repeat inside a split. It is not
        # a row identifier. Hashing the ID with deterministic row-index
        # tie breaking preserves the existing selection rule while
        # distinguishing legitimate rows with the same source ID.
        digest = int(hashlib.sha256(
            f"{seed}/{language}/{source_id}".encode("utf-8")
        ).hexdigest(), 16)
        entry = (-digest, -index, index, example)
        if len(selected) < count:
            heapq.heappush(selected, entry)
        elif entry[:2] > selected[0][:2]:
            heapq.heapreplace(selected, entry)
    if len(selected) != count:
        raise ValueError("Not enough official cases for " + language)
    return [
        {**entry[3], "_fleurs_source_row_index": entry[2]}
        for entry in sorted(selected, key=lambda entry: (-entry[0], -entry[1]))
    ]


def case_identity(language: str, split: str, source_row_index: int,
                  official_id: int) -> tuple[str, str]:
    """Define an immutable, collision-free identity for each source row."""
    if (language not in LANGUAGES or split not in ALLOWED_SPLITS or
            type(source_row_index) is not int or source_row_index < 0 or
            type(official_id) is not int or official_id < 0):
        raise ValueError("Invalid case identity")
    suffix = f"r{source_row_index:06d}_id{official_id}"
    return (
        f"fleurs_{split}_{language}_{suffix}",
        f"{language}_{suffix}.wav",
    )


def freeze_plan(destination: Path, split: str, count: int,
                seed: str, revision: str | None) -> tuple[str, Path]:
    """Pin the upstream revision before downloading or converting any WAV."""
    dest_plan = destination / "freeze_plan.json"
    manifest = destination / "manifest.json"
    if manifest.exists():
        raise FileExistsError("Completed frozen manifest already exists: " + str(manifest))
    expected = {
        "schema": PLAN_SCHEMA,
        "dataset": DATASET,
        "split": split,
        "per_language": count,
        "selection_seed": seed,
        "decoder_version": FREEZE_VERSION,
        "language_codes": list(LANGUAGES),
    }
    if dest_plan.exists():
        saved = json.loads(dest_plan.read_text(encoding="utf-8"))
        if (not isinstance(saved, dict) or
                any(saved.get(k) != value for k, value in expected.items()) or
                not isinstance(saved.get("dataset_revision"), str) or
                not SHA.fullmatch(saved["dataset_revision"]) or
                (revision and revision != saved["dataset_revision"])):
            raise ValueError(
                "Existing partial freeze has a different split, seed, "
                "decoder or revision; no mixed-version data permitted"
            )
        return saved["dataset_revision"], dest_plan
    if revision is None:
        from huggingface_hub import HfApi
        selected_revision = HfApi().dataset_info(DATASET).sha
    else:
        selected_revision = revision
    if not isinstance(selected_revision, str) or not SHA.fullmatch(selected_revision):
        raise ValueError("Pin an exact official dataset revision")
    expected["dataset_revision"] = selected_revision
    destination.mkdir(parents=True, exist_ok=True)
    with dest_plan.open("x", encoding="utf-8") as out:
        json.dump(expected, out, ensure_ascii=True, indent=2)
        out.write("\n")
    return selected_revision, dest_plan


def encode_pcm16(decoded) -> tuple[bytes, float]:
    import numpy as np
    if hasattr(decoded, "get_all_samples"):
        sample = decoded.get_all_samples()
        data, sr = sample.data.cpu().numpy(), sample.sample_rate
    else:
        data, sr = decoded["array"], decoded["sampling_rate"]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    if sr != 16000 or data.ndim != 1 or data.size == 0:
        raise ValueError("FLEURS sample must be 16kHz mono")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite audio")
    duration = len(data) / sr
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise ValueError("Audio sample duration outside published benchmark policy")
    pcm = (np.clip(data, -1, 1) * 32767).round().astype("<i2").tobytes()
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm)
    return stream.getvalue(), duration


def freeze_fleurs(root: Path, split: str, per_language: int, seed: str,
                  revision: str | None = None) -> tuple[dict, Path]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("Use validation for tuning or test for a sealed evaluation")
    if not 10 <= per_language <= 400:
        raise ValueError("Choose 10 to 400 cases per language")
    root = require_e_drive(root)
    configure_cache(root)
    from datasets import Audio, load_dataset
    # Separate output namespace avoids contaminating a partial freeze made
    # with an earlier TorchCodec or source revision. Never delete old WAVs.
    destination = root / (
        "fleurs_" + split + "_" + str(per_language) + "_" + FREEZE_VERSION
    )
    commit, plan_path = freeze_plan(
        destination, split, per_language, seed, revision
    )
    manifest = destination / "manifest.json"
    cases = []
    for lang, config in LANGUAGES.items():
        # Streaming only the requested split prevents eager downloads of
        # the entire train and test parquet sets for every language.
        dataset = load_dataset(
            DATASET, config, split=split, revision=commit,
            streaming=True, cache_dir=os.environ["HF_DATASETS_CACHE"]
        )
        # Keep compressed audio undecoded until selected. No TorchCodec.
        dataset = dataset.cast_column("audio", Audio(decode=False))
        chosen = choose_streaming_examples(
            dataset, language=lang, count=per_language, seed=seed
        )
        lang_cases = []
        for example in chosen:
            reference = str(example.get("transcription") or "").strip()
            if not reference:
                raise ValueError("Selected official reference is empty")
            pcm, duration = encode_pcm16(decode_audio_without_torchcodec(example["audio"]))
            source_row_index = example["_fleurs_source_row_index"]
            official_id = example["id"]
            case_id, filename = case_identity(
                lang, split, source_row_index, official_id
            )
            wav = destination / "wav" / filename
            wav.parent.mkdir(parents=True, exist_ok=True)
            hash_value = hashlib.sha256(pcm).hexdigest()
            if wav.exists():
                if hashlib.sha256(wav.read_bytes()).hexdigest() != hash_value:
                    raise ValueError(
                        "Existing WAV differs from pinned official decoding: "
                        + str(wav) + " existing_sha256="
                        + hashlib.sha256(wav.read_bytes()).hexdigest()
                        + " expected_sha256=" + hash_value
                        + ". No overwrite. Inspect this versioned directory."
                    )
            else:
                with wav.open("xb") as output:
                    output.write(pcm)
            lang_cases.append({
                "id": case_id,
                "source_row_index": source_row_index,
                "official_id": official_id,
                "official_audio_path": str(example.get("path") or ""),
                "language": lang,
                "corpus": "fleurs_published",
                "split": ALLOWED_SPLITS[split],
                "audio_path": str(wav.resolve()),
                "audio_sha256": hash_value,
                "audio_duration_seconds": round(duration, 4),
                "reference_text": reference,
                "reference_source": "google/fleurs@" + commit + "/" + config + "/" + split,
                "reference_kind": "published_transcript",
                "independently_audio_reviewed": False,
                "selected_independently_of_asr": True,
            })
        cases.extend(lang_cases)
        print("FROZEN", lang, len(lang_cases), "official cases", flush=True)
    document = {
        "schema": SCHEMA, "dataset": DATASET,
        "decoder_version": FREEZE_VERSION,
        "freeze_plan_file": str(plan_path.resolve()),
        "freeze_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "selected_split_streaming": True,
        "row_index_identity": True,
        "official_id_uniqueness_assumed": False,
        "dataset_revision": commit, "license": "CC BY 4.0",
        "attribution": "Google FLEURS",
        "data_split": split, "benchmark_role": ALLOWED_SPLITS[split],
        "selection_seed": seed, "per_language": per_language,
        "languages": list(LANGUAGES), "cases": cases,
        "reference_is_model_independent": True,
        "independently_listened_to": False,
        "source_evaluated_on_true_meeting_acoustics": False,
        "model_scores_used_to_choose_cases": False,
        "production_release_approved": False,
    }
    with manifest.open("x", encoding="utf-8") as out:
        json.dump(document, out, ensure_ascii=False, indent=2)
        out.write("\n")
    return document, manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="validation")
    p.add_argument("--per-language", type=int, default=20)
    p.add_argument("--seed", default="meeting-asr-selection-v1")
    p.add_argument("--revision", help="Optional pinned FLEURS dataset commit")
    p.add_argument("--apply", action="store_true",
                   help="Explicitly allow downloading FLEURS to E drive")
    args = p.parse_args(argv)
    try:
        if not args.apply:
            print("DRY RUN. No data downloaded. Pass --apply to fetch official",
                  args.split, "FLEURS data for", len(LANGUAGES), "languages.")
            return 0
        result, file = freeze_fleurs(args.root, args.split,
                                     args.per_language, args.seed, args.revision)
        print("FROZEN CORPUS", len(result["cases"]), "cases", file)
        print("MANUAL AUDIO REVIEW: NOT PERFORMED; PRODUCTION RELEASE: BLOCKED")
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print("FLEURS PREPARATION FAILED:", type(exc).__name__,
              str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

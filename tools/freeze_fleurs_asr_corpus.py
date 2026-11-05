"""Freeze independent Google FLEURS multilingual ASR cases to the E drive.

Optional network work is explicit. No model inference, no train split, no
selection on predictions, no automatic quality/pass labels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
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
SHA = re.compile(r"[0-9a-f]{64}")
MIN_SECONDS, MAX_SECONDS = 0.8, 30.0


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
    from datasets import load_dataset
    from huggingface_hub import HfApi
    commit = revision or HfApi().dataset_info(DATASET).sha
    if not isinstance(commit, str) or not SHA.fullmatch(commit):
        raise ValueError("Pin an exact 40/64 character dataset revision")
    destination = root / ("fleurs_" + split + "_" + str(per_language))
    manifest = destination / "manifest.json"
    if manifest.exists():
        raise FileExistsError("Existing corpus manifest is immutable: " + str(manifest))
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for lang, config in LANGUAGES.items():
        dataset = load_dataset(DATASET, config, split=split,
                               revision=commit,
                               cache_dir=os.environ["HF_DATASETS_CACHE"])
        # Select using IDs alone before any audio decoding or reference scoring.
        ids = [{"id": int(row["id"])} for row in dataset.select_columns(["id"])]
        chosen = choose_indices(ids, language=lang, count=per_language, seed=seed)
        if len(chosen) != per_language:
            raise ValueError("Not enough official cases for " + lang)
        lang_cases = []
        for index in chosen:
            example = dataset[index]
            reference = str(example.get("transcription") or "").strip()
            if not reference:
                raise ValueError("Selected official reference is empty")
            pcm, duration = encode_pcm16(example["audio"])
            sample_id = str(int(example["id"]))
            filename = lang + "_" + sample_id + ".wav"
            wav = destination / "wav" / filename
            wav.parent.mkdir(parents=True, exist_ok=True)
            hash_value = hashlib.sha256(pcm).hexdigest()
            if wav.exists():
                if hashlib.sha256(wav.read_bytes()).hexdigest() != hash_value:
                    raise ValueError("Frozen audio differs from official export: " + filename)
            else:
                with wav.open("xb") as output:
                    output.write(pcm)
            lang_cases.append({
                "id": "fleurs_" + split + "_" + lang + "_" + sample_id,
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
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print("FLEURS PREPARATION FAILED:", type(exc).__name__,
              str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

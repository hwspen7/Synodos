"""Prepare ungated authentic two-person speech from DataHive's open sample.

Four Fixed195 languages are covered: English, Hindi, Indonesian and Russian.
Audio-only collection is NOT a scored ASR benchmark: this source has no
verified transcript annotations. Do not create false transcript text or a
completed Fixed195 lock. Speaker identities are determined by original
stereo channels, not synthetic mixtures or a diarization model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fixed150_prepare import atomic_json, checksum, write_wav

REPO = "datahiveai/multilingual-conversations-sample"
TARGETS = {"en": 10, "hi": 5, "id": 5, "ru": 5}
SCHEMA = "meeting-fixed195-public-dialogue-audio-v1"
DURATION_SECONDS = 75
OFFSET_SECONDS = 15


def select_files(catalog: list[str], targets: dict[str, int] = TARGETS) -> dict[str, list[str]]:
    """Fail closed if a pinned source lacks genuine stereo conversations."""
    chosen = {}
    for language, required in targets.items():
        # Names documented in dataset card, one original conversation per MP3.
        candidates = sorted(
            path for path in catalog
            if path.startswith(f"preview/{language}_")
            and path.lower().endswith(".mp3")
            and "/" not in path[len("preview/"):]
        )
        if len(candidates) < required:
            raise ValueError(
                f"DataHive source has only {len(candidates)} {language} MP3s; "
                f"need {required}. Do not duplicate conversations."
            )
        chosen[language] = candidates[:required]
    return chosen


def _resample(x, rate: int):
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly
    samples = np.asarray(x, dtype=np.float32)
    if rate <= 0:
        raise ValueError("Invalid source sample rate")
    if rate != 16000:
        g = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // g, rate // g).astype(np.float32)
    return samples


def _active_fraction(x) -> float:
    """Detect activity on both genuine channels without claiming word labels."""
    import numpy as np
    window = 320  # 20ms
    n = len(x) // window
    if n < 100:
        return 0.0
    frames = x[:n * window].reshape(n, window)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.percentile(rms, 95))
    if peak < .002:
        return 0.0
    return float(np.mean(rms > max(.002, peak * .15)))


def convert_two_channels(source: Path, target: Path) -> dict:
    """Read a short crop of native dual-channel speech and preserve tracks."""
    import soundfile as sf
    import numpy as np

    with sf.SoundFile(str(source)) as stream:
        if stream.channels != 2 or stream.samplerate < 8000:
            raise ValueError(f"Expected independent stereo conversation: {source.name}")
        duration = stream.frames / stream.samplerate
        if duration < OFFSET_SECONDS + DURATION_SECONDS:
            raise ValueError(f"Original clip is too short: {source.name}")
        stream.seek(round(OFFSET_SECONDS * stream.samplerate))
        excerpt = stream.read(
            round(DURATION_SECONDS * stream.samplerate),
            dtype="float32", always_2d=True
        )
        rate = stream.samplerate

    if excerpt.shape[1] != 2 or not np.isfinite(excerpt).all():
        raise ValueError("Original stereo channels are missing or nonfinite")
    left, right = (_resample(excerpt[:, n], rate) for n in (0, 1))
    fractions = (_active_fraction(left), _active_fraction(right))
    if any(x < .025 for x in fractions):
        raise ValueError(
            f"Insufficient speech on both original tracks in {source.name}: "
            f"{fractions}"
        )
    if np.array_equal(left, right):
        raise ValueError("Source has duplicate channels, not two individual speakers")

    mixed = left + right
    peak = float(np.max(np.abs(mixed)))
    scale = min(1., .98 / peak) if peak else 1.
    target.parent.mkdir(parents=True, exist_ok=True)
    duration_s = write_wav(target, mixed * scale, 16000)
    left_path = target.with_name(target.stem + "_speaker1.wav")
    right_path = target.with_name(target.stem + "_speaker2.wav")
    write_wav(left_path, left * scale, 16000)
    write_wav(right_path, right * scale, 16000)
    return {
        "duration_s": duration_s,
        "stereo_source": True,
        "speaker_activity_fractions": [round(f, 4) for f in fractions],
        "mix_gain": round(scale, 6),
        "isolated_audio": [
            {"speaker": "speaker_1", "audio": left_path.name,
             "sha256": checksum(left_path)},
            {"speaker": "speaker_2", "audio": right_path.name,
             "sha256": checksum(right_path)},
        ],
    }


def prepare(root: Path, *, hf_home: str | None = None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    sha = api.dataset_info(REPO).sha
    listing = api.list_repo_files(repo_id=REPO, repo_type="dataset", revision=sha)
    selected = select_files(listing)
    checkpoint = root / "public_multi_progress.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA or state.get("source_sha") != sha:
            raise ValueError("Open source changed; use a new checkpoint directory")
    else:
        state = {"schema": SCHEMA, "source_repo": REPO,
                 "source_sha": sha, "cases": {}}
    for language, filenames in selected.items():
        for idx, filename in enumerate(filenames, 1):
            cid = f"fixed195-{language}-m{idx:02d}"
            target = root / "audio" / f"{cid}.wav"
            old = state["cases"].get(cid)
            if old:
                if old["source_file"] != filename or checksum(target) != old["sha256"]:
                    raise ValueError(f"Corrupt or changed public-audio checkpoint {cid}")
                continue
            try:
                downloaded = Path(hf_hub_download(
                    repo_id=REPO, repo_type="dataset", revision=sha,
                    filename=filename
                ))
            except GatedRepoError as exc:
                raise RuntimeError(
                    f"Unexpected access restriction on public dataset {REPO}."
                ) from exc
            geometry = convert_two_channels(downloaded, target)
            state["cases"][cid] = {
                "id": cid,
                "language": language,
                "kind": "multi",
                "scenario": "authentic_stereo_human_conversation",
                "audio": str(target.relative_to(root)).replace("\\", "/"),
                "sha256": checksum(target),
                "source_repo": REPO,
                "source_revision": sha,
                "source_file": filename,
                "source_original_sha256": checksum(downloaded),
                "source_excerpt_start_seconds": OFFSET_SECONDS,
                "source_excerpt_end_seconds": OFFSET_SECONDS + DURATION_SECONDS,
                "reference_status": "missing_human_transcript",
                "is_scoreable_asr": False,
                "is_scoreable_diarization": False,
                "speaker_channels": ["left", "right"],
                **geometry,
            }
            atomic_json(checkpoint, state)
            print(f"  {cid} public authentic stereo dialogue "
                  f"{geometry['duration_s']:.1f}s (transcript missing)", flush=True)
    cases = list(state["cases"].values())
    manifest = root / "public_multi_unscored.jsonl"
    manifest.write_text("".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in cases
    ), encoding="utf-8")
    count = len(cases)
    print(f"Public conversations collected: {count}/25; "
          "all 25 require human text/speaker timestamp annotation.")
    print("Other 40 conversations remain unsourced from this dataset.")
    print("Do NOT claim Fixed195 complete; the original 130 singles are unchanged.")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    args = parser.parse_args()
    prepare(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

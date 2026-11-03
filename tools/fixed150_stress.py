"""Create 40 deterministic acoustic stress variants, separate from 150 real/source cases.

Eight frozen single-speaker WAVs x five signal conditions. Synthetic variants
are deliberately NOT counted as independent speakers or new source recordings.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import wave

from tools.fixed150_prepare import verify_lock, checksum, write_wav, atomic_json

SOURCE_NUMBERS = (1, 13, 25, 37, 49, 57, 61, 65)
CONDITIONS = ("noise_15db", "noise_5db", "room_echo", "telephone_band", "low_volume")


def transform(audio, condition: str, seed: int):
    import numpy as np
    from scipy.signal import butter, sosfilt
    y = np.asarray(audio, dtype=np.float32).copy()
    if condition.startswith("noise_"):
        db = float(condition.split("_")[1].removesuffix("db"))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, len(y)).astype(np.float32)
        level = float(np.sqrt(np.mean(y * y) + 1e-12))
        noise /= float(np.sqrt(np.mean(noise * noise) + 1e-12))
        y += noise * level / (10 ** (db / 20))
    elif condition == "room_echo":
        # A deliberately synthetic sparse echo, not measured room impulse response.
        reflections = ((0.075, 0.38), (0.19, 0.23), (0.34, 0.11))
        original = y.copy()
        for delay, strength in reflections:
            n = round(16000 * delay)
            y[n:] += original[:-n] * strength
    elif condition == "telephone_band":
        y = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=16000,
                             output="sos"), y).astype(np.float32)
    elif condition == "low_volume":
        y *= 0.12
    else:
        raise ValueError(f"Unknown stress condition {condition}")
    return np.clip(y, -1.0, 1.0)


def build(root: Path) -> dict:
    import numpy as np
    lock = verify_lock(root)
    case_by_id = {c["id"]: c for c in lock["cases"]}
    target = root / "stress"
    target.mkdir(exist_ok=True)
    results = []
    for number in SOURCE_NUMBERS:
        original = case_by_id[f"fixed150-{number:03d}"]
        with wave.open(str(root / original["audio"]), "rb") as f:
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype("float32") / 32768
        for condition in CONDITIONS:
            case_id = f"stress-{number:03d}-{condition}"
            rel = f"stress/{case_id}.wav"
            path = root / rel
            seed = int(sha256(case_id.encode()).hexdigest()[:8], 16)
            if not path.exists():
                write_wav(path, transform(samples, condition, seed), 16000)
            result = {
                "id": case_id, "audio": rel, "reference": original["reference"],
                "language": original["language"], "language_mode": "auto",
                "base_id": original["id"], "base_sha256": original["sha256"],
                "condition": condition, "is_synthetic_augmentation": True,
                "sha256": checksum(path),
            }
            results.append(result)
    old = root / "stress_lock.json"
    payload = {"schema": "meeting-fixed150-stress-v1", "cases": results}
    if old.exists() and json.loads(old.read_text(encoding="utf-8")) != payload:
        raise ValueError("Stress WAV or base checksum changed; refuse to silently re-freeze")
    atomic_json(old, payload)
    (root / "manifest_stress.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in results), encoding="utf-8")
    print(f"Prepared {len(results)} correlated stress variants from {len(SOURCE_NUMBERS)} originals")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    args = parser.parse_args()
    build(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

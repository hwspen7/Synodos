"""Build traceable same-session verification trials from Fixed195 stereo stems.

This produces a DIAGNOSTIC, NOT a deployable speaker-verification test set.
The public multi-speaker collection stores two separated channel recordings
but does not contain independently annotated diarization references.
No model predictions are used as ground-truth speaker identities.

All generated WAVs and manifests live under the gitignored Fixed195 root.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import wave

import numpy as np

from tools.speaker_stage1_compare import MANIFEST_SCHEMA, verified_path

SOURCE_SCHEMA = "meeting-fixed195-public-dialogue-audio-v1"
TRIAL_SOURCE = "same_recording_isolated_channel_diagnostic_v1"
SAMPLE_RATE = 16000


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            SAMPLE_RATE, 1, 2
        ):
            raise ValueError("Require 16 kHz mono PCM16 WAV")
        data = wav.readframes(wav.getnframes())
    if not data:
        raise ValueError("Empty isolated speaker WAV")
    return np.frombuffer(data, dtype="<i2").copy()


def read_sources(path: Path, root: Path, case_id: str | None) -> list[dict]:
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source manifest must be inside the selected private corpus")
    entries = []
    ids = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"Malformed source manifest line {lineno}")
        key = case.get("id")
        if (not isinstance(key, str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", key)
                or key in ids):
            raise ValueError("Invalid or duplicate source case ID")
        ids.add(key)
        if case_id is not None and key != case_id:
            continue
        if (case.get("kind") != "multi"
                or case.get("scenario") != "authentic_stereo_human_conversation"
                or case.get("source_repo") is None
                or case.get("reference_status") != "missing_human_transcript"
                or case.get("is_scoreable_diarization") is not False
                or case.get("speaker_channels") != ["left", "right"]):
            raise ValueError(f"Unrecognized unscored stereo source: {key}")
        verified_path(root, case)
        isolated = case.get("isolated_audio")
        if (not isinstance(isolated, list) or len(isolated) != 2
                or {row.get("speaker") for row in isolated if isinstance(row, dict)}
                   != {"speaker_1", "speaker_2"}):
            raise ValueError(f"Missing isolated speaker source provenance: {key}")
        for row in isolated:
            name = row.get("audio")
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.lower().endswith(".wav")):
                raise ValueError("Isolated stem name must be a simple WAV filename")
            stem = {"audio": "audio/" + name, "sha256": row.get("sha256")}
            verified_path(root, stem)
        entries.append(case)
    if not entries:
        raise ValueError("No selected verified stereo-source records found")
    return entries


def window_candidates(samples: np.ndarray, *, clip_seconds: int,
                      min_active: float) -> list[tuple[float, int]]:
    """Rank non-overlapping full windows by actual channel activity.

    RMS is not proof of speech; every accepted stem must still be listened to.
    """
    size = SAMPLE_RATE * clip_seconds
    frame = SAMPLE_RATE // 5
    ranked = []
    for index in range(len(samples) // size):
        begin = index * size
        window = samples[begin:begin + size].astype(np.float32) / 32768.0
        frames = window.reshape(-1, frame)
        energy = np.sqrt(np.mean(frames * frames, axis=1))
        active_fraction = float(np.mean(energy > 0.008))
        overall_rms = float(np.sqrt(np.mean(window * window)))
        if active_fraction >= min_active and overall_rms >= 0.010:
            ranked.append((active_fraction + overall_rms, index))
    return sorted(ranked, key=lambda pair: (-pair[0], pair[1]))


def choose_windows(samples: np.ndarray, *, clip_seconds: int = 5,
                   min_active: float = 0.5) -> list[tuple[int, np.ndarray]]:
    if not 2 <= clip_seconds <= 15 or not 0 < min_active <= 1:
        raise ValueError("Invalid clip quality constraints")
    ranked = window_candidates(samples, clip_seconds=clip_seconds,
                               min_active=min_active)
    selected = []
    for _, idx in ranked:
        # Require a full unused window between same-speaker samples.
        if any(abs(idx - previous) < 2 for previous in selected):
            continue
        selected.append(idx)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        return []
    selected.sort()
    size = SAMPLE_RATE * clip_seconds
    return [(idx * clip_seconds, samples[idx * size:(idx + 1) * size])
            for idx in selected]


def write_pcm_exclusive(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        # wave.open accepts a file object; use exclusive mode to avoid
        # overwriting existing licensed/private benchmark audio.
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.astype("<i2", copy=False).tobytes())


def build(root: Path, source_manifest: Path, output: Path, *,
          case_id: str | None = None, clip_seconds: int = 5,
          min_active: float = 0.5) -> dict:
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root) or output.suffix.lower() != ".json":
        raise ValueError("Output JSON must remain under the private benchmark root")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing trial manifest")
    entries = read_sources(source_manifest.resolve(), root, case_id)
    selected = []
    skipped = []
    # Validate every candidate before creating ANY output clip.
    for case in entries:
        windows = {}
        for stem in case["isolated_audio"]:
            descriptor = {"audio": "audio/" + stem["audio"],
                          "sha256": stem["sha256"]}
            samples = read_pcm(verified_path(root, descriptor))
            chunks = choose_windows(samples, clip_seconds=clip_seconds,
                                    min_active=min_active)
            if len(chunks) != 2:
                windows = {}
                break
            windows[stem["speaker"]] = chunks
        if windows:
            selected.append((case, windows))
        else:
            skipped.append(case["id"])
    if not selected:
        raise ValueError(
            "No case has two sufficiently active independent windows for "
            "BOTH speakers. Check isolated tracks and quality constraints."
        )
    new_files = []
    pairs = []
    try:
        for case, windows in selected:
            descriptors = {}
            for speaker in ("speaker_1", "speaker_2"):
                for index, (start_s, pcm) in enumerate(windows[speaker], 1):
                    relative = (
                        "speaker_stage1_samples/" + case["id"] + "/" +
                        speaker + "_part" + str(index) + ".wav"
                    )
                    dest = (root / relative).resolve()
                    if not dest.is_relative_to(root):
                        raise ValueError("Generated clip path escaped corpus root")
                    write_pcm_exclusive(dest, pcm)
                    new_files.append(dest)
                    descriptors[(speaker, index)] = {
                        "audio": relative.replace("\\", "/"),
                        "sha256": digest(dest),
                    }
            # Same-speaker pair uses disjoint 5-second portions of ONE
            # isolated channel. Different-speaker pair uses separate channels
            # from the same conversation. Both are same-session diagnostics.
            for index, speaker in enumerate(("speaker_1", "speaker_2"), 1):
                pairs.append({
                    "id": case["id"] + "-same-" + str(index),
                    "left": descriptors[(speaker, 1)],
                    "right": descriptors[(speaker, 2)],
                    "same_speaker": True,
                })
            for index in (1, 2):
                pairs.append({
                    "id": case["id"] + "-different-" + str(index),
                    "left": descriptors[("speaker_1", index)],
                    "right": descriptors[("speaker_2", index)],
                    "same_speaker": False,
                })
        result = {
            "schema": MANIFEST_SCHEMA,
            "diarization_cases": [],
            "verification_pairs": pairs,
            "verification_provenance": {
                "kind": TRIAL_SOURCE,
                "source_manifest_sha256": digest(source_manifest),
                "paired_recordings": len(selected),
                "skipped_recording_ids": skipped,
                "clip_seconds": clip_seconds,
                "requires_manual_stem_review": True,
                "cross_session_identity_evaluable": False,
                "operational_threshold_evaluable": False,
                "warning": (
                    "Same-speaker pairs are from ONE conversation and "
                    "cannot establish cross-meeting identification accuracy. "
                    "Channel isolation does not guarantee no bleed or noise."
                ),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except Exception:
        for file in new_files:
            file.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed195"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--clip-seconds", type=int, default=5)
    parser.add_argument("--min-active", type=float, default=0.5)
    args = parser.parse_args()
    root = args.root
    source = args.source or root / "public_multi_unscored.jsonl"
    output = args.output or root / "speaker_stage1_manifest.json"
    data = build(root, source, output, case_id=args.case_id,
                 clip_seconds=args.clip_seconds, min_active=args.min_active)
    print(json.dumps({
        "manifest": str(output),
        "paired_recordings": data["verification_provenance"]["paired_recordings"],
        "skipped_recording_ids":
            data["verification_provenance"]["skipped_recording_ids"],
        "same_person_pairs": len(data["verification_pairs"]) // 2,
        "different_person_pairs": len(data["verification_pairs"]) // 2,
        "cross_session_identity_evaluable": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

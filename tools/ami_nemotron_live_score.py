"""Score a full-session independent Nemotron stream against frozen AMI RTTM.

Unlike the older 2x120-second offline comparison, this tests ONE continuous
240-second stream and a speaker cache that survives all chunks. Standard DER
includes overlaps. An additional solo-only diagnostic is clearly separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.ami_shadow10_prepare import SELECTED
from tools.ami_shadow10_score import score_case
from tools.phase09_speaker_benchmark import evaluate
from tools.speaker_stage1_compare import _reference_turns, diarization_details


def _validated_independent(replay: dict) -> list[dict]:
    acoustic = replay.get("independent_diarization")
    if not isinstance(acoustic, dict):
        raise ValueError("Independent Nemotron track missing")
    if (acoustic.get("backend") != "nemotron3"
            or acoustic.get("complete") is not True
            or acoustic.get("state") != "completed"
            or acoustic.get("error")
            or acoustic.get("cross_chunk_identity_cache") is not True
            or acoustic.get("frame_step_ms") != 10):
        raise ValueError("Nemotron track incomplete or cache continuity unverified")
    source_samples = replay.get("sent_samples")
    if (type(source_samples) is not int or
            acoustic.get("processed_audio_samples") != source_samples):
        raise ValueError("Independent diarization did not consume all source PCM")
    frames = acoustic.get("scored_frames")
    if (type(frames) is not int or frames < 0
            or abs(frames * 160 - source_samples) > 8000):
        raise ValueError("Nemotron frame coverage does not match recorded audio")
    rows = acoustic.get("turns")
    if not isinstance(rows, list):
        raise ValueError("Nemotron speaker turns missing")
    checked = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed Nemotron speaker turn")
        a, b, spk = (row.get("start_ms"), row.get("end_ms"),
                     row.get("speaker_id"))
        if (type(a) is not int or type(b) is not int
                or type(spk) is not int or not 1 <= spk <= 8
                or a < 0 or b <= a or b > source_samples * 1000 // 16000):
            raise ValueError("Nemotron turn violates source bounds")
        checked.append({
            "start_ms": a, "end_ms": b, "speaker_id": spk,
        })
    return checked


def _solo_only(reference: list[dict], predicted: list[dict],
               *, duration_ms: int, step_ms: int = 100) -> dict:
    """Evaluate ONLY frames with exactly one human reference talker.

    Silence and overlapping-reference frames are excluded on both sides.
    This is intentionally NOT the standard DER and must not be described as
    a model's headline DER or directly compared to official AMI numbers.
    """
    refs, hyps = [], []
    for t in range(0, duration_ms, step_ms):
        real = {
            str(item["speaker"]) for item in reference
            if item["start_ms"] <= t < item["end_ms"]
        }
        if len(real) != 1:
            continue
        refs.append({
            "start_ms": t, "end_ms": t + step_ms,
            "speaker": next(iter(real)),
        })
        for speaker in {
            row["speaker_id"] for row in predicted
            if row["start_ms"] <= t < row["end_ms"]
        }:
            hyps.append({
                "start_ms": t, "end_ms": t + step_ms,
                "speaker_id": speaker,
            })
    if not refs:
        raise ValueError("No solo speaker reference frames")
    from tools.speaker_stage1_compare import _score_diarization
    return _score_diarization(refs, hyps)


def score_independent(root: Path, meeting: str, replay_path: Path,
                      baseline_path: Path | None = None) -> dict:
    # Verify the pin, WAV checksum, RTTM checksum and transcript timeline
    # before examining a purported independent track.
    caption_audit = score_case(root, meeting, replay_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay.get("server_ready", {}).get("independent_diarization") != "nemotron3":
        raise ValueError("Replay was not recorded with Nemotron enabled")
    independent = _validated_independent(replay)
    case = json.loads((root / "cases" / f"{meeting}.json").read_text(encoding="utf-8"))
    ref = _reference_turns(case["reference_turns"], 240000)
    raw = evaluate(ref, independent, step_ms=100)
    details = diarization_details(ref, independent, raw)
    raw.pop("speaker_mapping", None)
    solo = _solo_only(ref, independent, duration_ms=240000)
    result = {
        "schema": "meeting-ami-independent-nemotron-live-score-v1",
        "meeting": meeting,
        "source_audio_sha256": caption_audit["source_audio_sha256"],
        "reference_source_sha256": caption_audit["reference_source_sha256"],
        "replay_sha256": caption_audit["replay_sha256"],
        "continuous_stream": True,
        "model": "nvidia/Nemotron-3-Diarization",
        "independent_acoustic_metrics": {**raw, **details},
        "solo_speech_only_metrics": solo,
        "published_caption_metrics": caption_audit["metrics"],
        "caption_gap_seconds": caption_audit[
            "reference_speech_without_captions"
        ]["reference_speech_without_captions_seconds"],
        "under_10pct_standard_der": raw["der"] < 0.10,
        "under_20pct_standard_der": raw["der"] < 0.20,
        "limitations": (
            "Full 240-second source-independent stream, identical frozen AMI "
            "only_words human RTTM and 100ms zero-collar overlap-inclusive "
            "scorer. Not the exact NVIDIA public evaluation protocol. "
            "Solo-only DER excludes both silence and overlapping-reference "
            "frames, and is NOT standard DER. Subtitle correctness requires "
            "independent transcript reference/WER review."
        ),
    }
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (baseline.get("source_audio_sha256")
                != result["source_audio_sha256"]
                or baseline.get("reference_source_sha256")
                != result["reference_source_sha256"]):
            raise ValueError("Baseline and Nemotron refer to different sources")
        base = float(baseline["metrics"]["der"])
        result["comparison"] = {
            "baseline_caption_derived_der": base,
            "independent_acoustic_der": raw["der"],
            "delta": round(raw["der"] - base, 5),
            "note": (
                "Both share the same reference and scoring definition, "
                "but one uses ASR-timed captions and the other independently "
                "measures acoustic speaker activity."
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("benchmark/fixed195/ami_shadow10"))
    parser.add_argument("--meeting", required=True,
                        choices=[name for name, _ in SELECTED])
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-under-10", action="store_true")
    parser.add_argument("--require-under-20", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if (not root.is_relative_to(Path("benchmark/fixed195").resolve())
            or not output.is_relative_to(root) or output.suffix != ".json"
            or output.exists()):
        parser.error("Output must be new JSON inside private Fixed195 root")
    report = score_independent(root, args.meeting, args.replay,
                               args.baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "meeting": args.meeting,
        "independent_acoustic_metrics": report["independent_acoustic_metrics"],
        "solo_speech_only_metrics": report["solo_speech_only_metrics"],
        "published_caption_metrics": report["published_caption_metrics"],
        "caption_gap_seconds": report["caption_gap_seconds"],
        "under_10pct_standard_der": report["under_10pct_standard_der"],
        "under_20pct_standard_der": report["under_20pct_standard_der"],
        "comparison": report.get("comparison"),
        "output": str(output),
    }, indent=2))
    if args.require_under_10 and not report["under_10pct_standard_der"]:
        return 1
    if args.require_under_20 and not report["under_20pct_standard_der"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

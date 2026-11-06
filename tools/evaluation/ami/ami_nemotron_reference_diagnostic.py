"""Diagnostic reference-label comparison using NVIDIA-linked AMI RTTM.

The frozen project only_words result is NEVER overwritten or replaced.
This computes a separate DER under the forced-aligned reference protocol
linked by NVIDIA's Nemotron 3 model card.

Changing ground truth does not improve the model. Both numbers remain visible.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from tools.evaluation.ami.ami_shadow10_prepare import SELECTED, checksum
from tools.evaluation.ami.ami_nemotron_live_score import _validated_independent, _solo_only
from tools.evaluation.ami.ami_shadow10_score import score_case
from tools.validation.dialogue_speaker_benchmark import evaluate
from tools.evaluation.speakers.speaker_stage1_compare import diarization_details, _reference_turns, baseline_turns
from speech.captions.asr_stage_trace import merged_intervals

REFERENCE_COMMIT = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
REFERENCE_SOURCE = "nttcslab-sp/diar-forced-alignment"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    + REFERENCE_SOURCE + "/" + REFERENCE_COMMIT + "/AMI/test/"
)
DURATION_MS = 240000


def forced_reference(raw: str, meeting: str, offset_s: int) -> list[dict]:
    """Validate and clip the independently sourced forced-alignment RTTM."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty forced RTTM")
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if (len(fields) != 10 or fields[0] != "SPEAKER"
                or fields[1] != meeting or fields[2] != "1"
                or not re.fullmatch(
                    re.escape(meeting) + r"\.[A-Za-z0-9]+", fields[7]
                )):
            raise ValueError("Unexpected forced RTTM format or speaker")
        try:
            absolute_s = float(fields[3])
            length_s = float(fields[4])
        except ValueError as exc:
            raise ValueError("Non-numeric forced RTTM time") from exc
        if (not 0 <= absolute_s <= 100000 or
                not 0 < length_s <= 120 or
                absolute_s + length_s > 100000):
            raise ValueError("Invalid forced RTTM duration")
        a = max(0, round((absolute_s - offset_s) * 1000))
        b = min(DURATION_MS, round((absolute_s + length_s - offset_s) * 1000))
        if b > a:
            rows.append({
                "start_ms": a, "end_ms": b, "speaker": fields[7],
            })
    if not rows:
        raise ValueError("No forced-reference speech in source excerpt")
    return sorted(rows, key=lambda x: (
        x["start_ms"], x["end_ms"], x["speaker"],
    ))


def fetch_forced_rttm(root: Path, meeting: str) -> tuple[str, str, Path]:
    """Fetch only pinned public reference labels to private benchmark root.

    No OS temporary folder, no Hugging Face default C-drive caches, and no
    replacement of a prior file or of the user's original AMI annotations.
    """
    filename = meeting + "-" + REFERENCE_COMMIT[:12] + ".rttm"
    dest = root / "forced_aligned_reference" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        request = Request(BASE_URL + meeting + ".rttm", headers={
            "User-Agent": "Meeting-AMI-DER-Reference-Check/1.0",
        })
        with urlopen(request, timeout=45) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) <= 100 or len(raw) > 1024 * 1024:
            raise ValueError("Unexpected forced RTTM download length")
        with dest.open("xb") as out:
            out.write(raw)
    raw = dest.read_bytes()
    if len(raw) <= 100 or len(raw) > 1024 * 1024:
        raise ValueError("Invalid cached forced RTTM length")
    return raw.decode("utf-8"), sha256(raw).hexdigest(), dest


def acoustic_turns_within_caption_clock(
    independent: list[dict], replay: dict, *, duration_ms: int,
) -> list[dict]:
    """Constrain raw acoustic activity to clocks with a final subtitle.

    Diagnostic counterfactual only. This does not restore missing text or
    assert that the transcript contains the right words.
    """
    timeline = (replay.get("final_cluster_diagnostics") or {}).get(
        "anonymous_timeline"
    )
    if not isinstance(timeline, dict) or timeline.get("truncated") is not False:
        raise ValueError("Complete published caption timeline required")
    rows = timeline.get("segments")
    if (not isinstance(rows, list)
            or timeline.get("total_segments") != len(rows)):
        raise ValueError("Incomplete published caption timeline")
    clock = merged_intervals(
        ((row["start_ms"], row["end_ms"]) for row in rows),
        duration_ms=duration_ms,
    )
    clipped = []
    for speaker in independent:
        for a, b in clock:
            start = max(a, speaker["start_ms"])
            end = min(b, speaker["end_ms"])
            if end > start:
                clipped.append({
                    "start_ms": start, "end_ms": end,
                    "speaker_id": speaker["speaker_id"],
                })
    return clipped


def display_projection_turns(replay: dict, duration_ms: int) -> tuple[list[dict], int]:
    """Diagnostic for actual UI speaker cues, NOT canonical stored caption DER.

    If the feature is off, raise rather than silently reporting the original
    caption metric as a projection gain. Unsplit captions retain original
    speaker IDs. No words, text, or audio are changed.
    """
    timeline = (replay.get("final_cluster_diagnostics") or {}).get(
        "anonymous_timeline"
    )
    if (not isinstance(timeline, dict) or
            timeline.get("truncated") is not False or
            timeline.get("total_segments") != len(timeline.get("segments", []))):
        raise ValueError("Complete anonymous caption timeline is required")
    turns = []
    split_count = 0
    for row in timeline["segments"]:
        cues = row.get("caption_projection_turns") or []
        status = row.get("caption_projection_status")
        if status == "split_verified" and len(cues) > 1:
            for cue in cues:
                a, b, speaker = (
                    cue.get("start_ms"), cue.get("end_ms"),
                    cue.get("speaker_id"),
                )
                if (type(a) is not int or type(b) is not int or
                        type(speaker) is not int or a < 0 or
                        b <= a or b > duration_ms):
                    raise ValueError("Invalid projected caption cue")
                if speaker > 0:
                    turns.append({
                        "start_ms": a, "end_ms": b, "speaker_id": speaker,
                    })
            split_count += 1
        else:
            a, b, speaker = (
                row.get("start_ms"), row.get("end_ms"),
                row.get("published_speaker_id"),
            )
            if type(speaker) is int and speaker > 0:
                turns.append({
                    "start_ms": a, "end_ms": b, "speaker_id": speaker,
                })
    return turns, split_count


def alternate_score(root: Path, meeting: str, replay_path: Path) -> dict:
    if meeting not in {name for name, _ in SELECTED}:
        raise ValueError("Meeting not in frozen AMI inventory")
    # First enforce the original audio checksum and original RTTM pin.
    original = score_case(root, meeting, replay_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay.get("server_ready", {}).get("independent_diarization") != "nemotron3":
        raise ValueError("This replay was not an independent Nemotron run")
    prediction = _validated_independent(replay)
    frozen_case = json.loads(
        (root / "cases" / (meeting + ".json")).read_text(encoding="utf-8")
    )
    old_reference = _reference_turns(
        frozen_case["reference_turns"], DURATION_MS,
    )
    old_metric = evaluate(old_reference, prediction, step_ms=100)
    old_details = diarization_details(old_reference, prediction, old_metric)
    old_metric.pop("speaker_mapping", None)
    raw, reference_sha, reference_file = fetch_forced_rttm(root, meeting)
    offset = dict(SELECTED)[meeting]
    reference = forced_reference(raw, meeting, offset)
    measured = evaluate(reference, prediction, step_ms=100)
    details = diarization_details(reference, prediction, measured)
    measured.pop("speaker_mapping", None)
    # Measure the actual final published captions under the SAME forced
    # reference, not only the independent raw speaker stream. This isolates
    # audio-model quality from text-alignment/fusion degradation.
    caption_prediction = baseline_turns(replay, duration_ms=DURATION_MS)
    caption_measured = evaluate(reference, caption_prediction, step_ms=100)
    caption_details = diarization_details(
        reference, caption_prediction, caption_measured,
    )
    caption_measured.pop("speaker_mapping", None)
    masked_prediction = acoustic_turns_within_caption_clock(
        prediction, replay, duration_ms=DURATION_MS,
    )
    masked = evaluate(reference, masked_prediction, step_ms=100)
    masked_details = diarization_details(reference, masked_prediction, masked)
    masked.pop("speaker_mapping", None)
    solo = _solo_only(reference, prediction, duration_ms=DURATION_MS)
    projected_turns, projected_split_count = display_projection_turns(
        replay, DURATION_MS,
    )
    if projected_split_count:
        projected = evaluate(reference, projected_turns, step_ms=100)
        projected.update(diarization_details(
            reference, projected_turns, projected,
        ))
        projected.pop("speaker_mapping", None)
    else:
        projected = None
    return {
        "schema": "meeting-ami-forced-reference-diagnostic-v1",
        "meeting": meeting,
        "audio_sha256": original["source_audio_sha256"],
        "original_reference_sha256": original["reference_source_sha256"],
        "alternate_reference_sha256": reference_sha,
        "alternate_reference_path": str(reference_file),
        "alternate_reference_origin": REFERENCE_SOURCE,
        "alternate_reference_commit": REFERENCE_COMMIT,
        "same_audio_and_prediction": True,
        "different_reference_annotations": True,
        "original_only_words_independent_der": old_metric["der"],
        "original_only_words_independent_metrics": {**old_metric, **old_details},
        "original_only_words_caption_der": original["metrics"]["der"],
        "forced_aligned_independent_der": measured["der"],
        "forced_aligned_independent_metrics": {**measured, **details},
        "forced_aligned_caption_clock_masked_acoustic_der": masked["der"],
        "forced_aligned_caption_clock_masked_acoustic_metrics": {
            **masked, **masked_details,
        },
        "caption_clock_masked_acoustic_note": (
            "Diagnostic counterfactual: independent Nemotron speech turns "
            "restricted to the union of final caption time windows, "
            "including unknown-speaker captions. It is NOT published-caption "
            "DER, a valid text recovery, or a replacement for live results."
        ),
        "forced_aligned_caption_der": caption_measured["der"],
        "forced_aligned_caption_metrics": {
            **caption_measured, **caption_details,
        },
        "forced_aligned_solo_only_metrics": solo,
        "display_speaker_cue_splits": projected_split_count,
        "forced_aligned_display_projection_metrics": projected,
        "display_projection_note": (
            "Separate post-finalization UI projection of confirmed ASR words "
            "against official Nemotron speaker turns. It is NOT canonical "
            "published-caption DER, does not recover missing words, and "
            "cannot by itself supply two simultaneous text transcripts. "
            "Null means no verified speaker-switch cue splitting was emitted."
        ),
        "note": (
            "This uses different ground truth and cannot be called an "
            "improvement to the old metric, a model enhancement, or a "
            "verified reproduction of the full NVIDIA AMI test benchmark. "
            "The original only_words DER remains immutable. "
            "Model acceptance must use a frozen, separately held-out protocol."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "benchmark/fixed195/ami_shadow10"
    ))
    parser.add_argument("--meeting", required=True,
                        choices=[name for name, _ in SELECTED])
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output.resolve()
    private = Path("benchmark/fixed195").resolve()
    if (not root.is_relative_to(private) or
            not destination.is_relative_to(root)
            or destination.suffix != ".json"
            or destination.exists()):
        parser.error("Choose a NEW JSON output inside the private Fixed195 root")
    result = alternate_score(root, args.meeting, args.replay)
    with destination.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

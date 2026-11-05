"""Prepare a balanced, private audio-listening queue from verified AMI recordings.

The queue is a review aid, never independently verified audio truth, never
a fabricated QA gold manifest, and never a product release acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import wave

from tools.evaluate_meeting_audio_quality import (
    evaluate_windows, tokenize, verify_reference,
)
from tools.prepare_meeting_reference_corpus import ASSETS, MEETINGS, sha256_file
from tools.prepare_real_meeting_evaluation import (
    DEFAULT_ROOT, check_root, load_predictions,
)

BUCKET_ORDER = (
    "CLEAN_AGREEMENT", "CLEAN_DISAGREEMENT", "CROSS_TALK",
    "POSSIBLE_DELETION", "POSSIBLE_INSERTION",
)
BUCKET_QUOTAS = {
    "CLEAN_AGREEMENT": 3,
    "CLEAN_DISAGREEMENT": 3,
    "CROSS_TALK": 3,
    "POSSIBLE_DELETION": 2,
    "POSSIBLE_INSERTION": 1,
}
FIELDS = (
    "meeting_id", "category", "start_s", "end_s", "clip_path",
    "clip_sha256", "source_audio_sha256", "reference_words",
    "asr_words", "reference_overlap_fraction", "screening_wer",
    "asr_segment_ids", "official_reference_text", "whisper_prediction_text",
    "audio_review_status", "reviewer", "reviewer_notes",
)


def word_overlap_indices(words):
    """Identify annotated lexical intervals intersecting another speaker.

    Sweep by start time and avoid guessing an interval for zero-duration words.
    A flagged reference word is not necessarily audible in a mono mix.
    """
    active = []
    affected = set()
    ordered = sorted(enumerate(words), key=lambda x: (x[1]["start_ms"], x[1]["end_ms"]))
    for idx, word in ordered:
        start = word["start_ms"]
        active = [(j, previous) for j, previous in active
                  if previous["end_ms"] > start]
        if word["end_ms"] <= start:
            continue
        for j, previous in active:
            if previous["speaker"] != word["speaker"]:
                affected.add(idx)
                affected.add(j)
        active.append((idx, word))
    return affected


def annotation_stats(words, flagged):
    total = sum(len(tokenize(w["text"])) for w in words)
    overlap = sum(len(tokenize(words[i]["text"])) for i in flagged)
    zero = sum(len(tokenize(w["text"])) for w in words
               if w["end_ms"] == w["start_ms"])
    isolated = total - overlap - zero
    if isolated < 0:
        raise ValueError("Invalid overlap/zero-duration token accounting")
    return {
        "annotated_reference_tokens": total,
        "other_speaker_overlap_tokens": overlap,
        "zero_duration_reference_tokens": zero,
        "nonoverlap_nonzero_reference_tokens": isolated,
        "other_speaker_overlap_fraction": round(overlap / total, 5) if total else 0.0,
        "nonoverlap_nonzero_fraction": round(isolated / total, 5) if total else 0.0,
    }


def classify_window(row, fraction):
    r, p = row["reference_words"], row["predicted_words"]
    wer = row["screening_wer"]
    if fraction >= .05:
        return "CROSS_TALK"
    if r >= 5 and p <= 2:
        return "POSSIBLE_DELETION"
    if p >= 5 and r <= 2:
        return "POSSIBLE_INSERTION"
    if r >= 5 and p >= 4 and fraction <= .01:
        return "CLEAN_AGREEMENT" if wer is not None and wer <= .35 else "CLEAN_DISAGREEMENT"
    return "CONTEXT_REVIEW"


def rank_row(row):
    group = row["category"]
    if group == "CLEAN_AGREEMENT":
        return (row["screening_wer"] if row["screening_wer"] is not None else 5,
                -row["reference_words"], row["start_s"])
    if group == "CLEAN_DISAGREEMENT":
        return (-min(3, row["screening_wer"] or 0),
                -row["reference_words"], row["start_s"])
    if group == "CROSS_TALK":
        return (-row["reference_overlap_fraction"], -row["reference_words"],
                row["start_s"])
    return (-max(row["reference_words"], row["predicted_words"]), row["start_s"])


def select_windows(rows, count=12, min_spacing_seconds=20):
    """Balanced deterministic selection, with a fallback for sparse buckets."""
    if count < 1:
        raise ValueError("Selection count must be positive")
    buckets = {name: [] for name in (*BUCKET_ORDER, "CONTEXT_REVIEW")}
    for row in rows:
        buckets[row["category"]].append(row)
    for group in buckets:
        buckets[group].sort(key=rank_row)

    selected = []
    used = set()

    def add(group, quota, spacing):
        for item in buckets[group]:
            if len([r for r in selected if r["category"] == group]) >= quota:
                break
            marker = (item["meeting_id"], item["start_s"])
            if marker in used:
                continue
            if any(abs(item["start_s"] - prior["start_s"]) < spacing
                   for prior in selected):
                continue
            used.add(marker)
            selected.append(item)

    for group in BUCKET_ORDER:
        add(group, min(count, BUCKET_QUOTAS[group]), min_spacing_seconds)

    while len(selected) < min(count, len(rows)):
        previous = len(selected)
        for group in (*BUCKET_ORDER, "CONTEXT_REVIEW"):
            add(group, count, min_spacing_seconds)
            if len(selected) >= count:
                break
        if len(selected) == previous:
            if min_spacing_seconds:
                min_spacing_seconds = 0
            else:
                break
    return sorted(selected[:count], key=lambda r: r["start_s"])


def make_review_rows(meeting, duration, words, segments, window_seconds):
    _, windows = evaluate_windows(meeting, duration, words, segments, window_seconds)
    flagged = word_overlap_indices(words)
    indexed = defaultdict(list)
    for i, word in enumerate(words):
        midpoint = (word["start_ms"] + word["end_ms"]) / 2000
        index = min(len(windows) - 1, int(midpoint // window_seconds))
        indexed[index].append(i)
    for j, row in enumerate(windows):
        indices = indexed[j]
        tokens = sum(len(tokenize(words[i]["text"])) for i in indices)
        flagged_tokens = sum(len(tokenize(words[i]["text"])) for i in indices
                             if i in flagged)
        fraction = flagged_tokens / tokens if tokens else 0
        row["meeting_id"] = meeting
        row["reference_overlap_fraction"] = round(fraction, 5)
        row["category"] = classify_window(row, fraction)
    return windows, annotation_stats(words, flagged)


def extract_wav_clip(wav_path, output, start_seconds, end_seconds):
    if not start_seconds < end_seconds:
        raise ValueError("Invalid clip time bounds")
    with wave.open(str(wav_path), "rb") as source:
        if (source.getframerate(), source.getnchannels(),
                source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Unexpected source WAV encoding")
        start = max(0, round(start_seconds * 16000))
        end = min(source.getnframes(), round(end_seconds * 16000))
        if not 0 <= start < end:
            raise ValueError("Empty WAV extraction")
        source.setpos(start)
        frames = source.readframes(end - start)
    if len(frames) != (end - start) * 2:
        raise ValueError("Truncated source WAV")
    with wave.open(str(output), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(frames)
    return round((end - start) / 16000, 3)


def create_queue(root, output_dir, per_meeting=12, window_seconds=10, pad_seconds=1.0):
    if not 4 <= per_meeting <= 30:
        raise ValueError("Select 4-30 clips per meeting")
    if not 5 <= window_seconds <= 30:
        raise ValueError("Use 5-30 second review windows")
    if not 0 <= pad_seconds <= 3:
        raise ValueError("Padding must be 0-3 seconds")
    archive = root / "annotations/ami_public_manual_1.6.2.zip"
    expected_archive = dict(ASSETS)["annotations/ami_public_manual_1.6.2.zip"]
    if not archive.is_file() or sha256_file(archive) != expected_archive:
        raise ValueError("Official AMI annotations missing or changed")

    predictions = load_predictions(root)
    if {row[0] for row in predictions} != set(MEETINGS):
        raise ValueError("Three distinct AMI predictions required")

    prepared = []
    for mid, wav_path, duration, segments, _ in predictions:
        audio_sha = sha256_file(wav_path)
        reference_file = root / "reference" / (mid + "_official_words.json")
        if not reference_file.is_file():
            raise FileNotFoundError("Generate official reference first: " + str(reference_file))
        words = verify_reference(json.loads(reference_file.read_text(encoding="utf-8")),
                                 mid, audio_sha, expected_archive, duration)
        windows, annotation = make_review_rows(mid, duration, words, segments,
                                               window_seconds)
        chosen = select_windows(windows, per_meeting)
        prepared.append((mid, wav_path, duration, audio_sha, annotation, chosen))

    if output_dir.exists():
        raise FileExistsError("Review output already exists; never overwrite")
    output_dir.mkdir(parents=True, exist_ok=False)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir()
    all_review_rows, totals = [], {}
    for mid, wav_path, duration, audio_sha, annotation, windows in prepared:
        for row in windows:
            start = max(0.0, row["start_s"] - pad_seconds)
            end = min(duration, row["end_s"] + pad_seconds)
            filename = mid + "_" + str(round(start * 1000)) + "_" + str(round(end * 1000)) + ".wav"
            path = clips_dir / filename
            clip_duration = extract_wav_clip(wav_path, path, start, end)
            if clip_duration <= 0:
                raise ValueError("Extracted an empty clip")
            all_review_rows.append({
                "meeting_id": mid, "category": row["category"],
                "start_s": round(start, 3), "end_s": round(end, 3),
                "clip_path": str(path),
                "clip_sha256": sha256_file(path),
                "source_audio_sha256": audio_sha,
                "reference_words": row["reference_words"],
                "asr_words": row["predicted_words"],
                "reference_overlap_fraction": row["reference_overlap_fraction"],
                "screening_wer": row["screening_wer"],
                "asr_segment_ids": row["asr_segment_ids"],
                "official_reference_text": row["official_reference_text"],
                "whisper_prediction_text": row["whisper_prediction_text"],
                "audio_review_status": "UNREVIEWED",
                "reviewer": "", "reviewer_notes": "",
            })
        totals[mid] = {**annotation, "selected_clip_count": len(windows),
                       "selected_categories": dict(Counter(r["category"] for r in windows)),
                       "source_audio_sha256": audio_sha}

    csv_path = output_dir / "review_queue.csv"
    with csv_path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_review_rows)
    result = {
        "schema": "meeting-audio-listening-queue",
        "source": "AMI Consortium manual words, CC BY 4.0",
        "status": "UNREVIEWED",
        "independent_audio_gold": False,
        "knowledge_release_pass": False,
        "source_model": "speech.whisper.WhisperEngine",
        "selected_clips": len(all_review_rows),
        "window_seconds": window_seconds,
        "padding_seconds": pad_seconds,
        "meetings": totals,
        "limitations": [
            "Official word time-overlap does not prove intelligibility from the mono mix.",
            "Screening WER is biased by reference time bucket and ASR utterance boundaries.",
            "Review these original-audio clips before creating independent QA gold.",
        ],
    }
    with (output_dir / "queue_inventory.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result, csv_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--per-meeting", type=int, default=12)
    p.add_argument("--window-seconds", type=int, default=10)
    p.add_argument("--pad-seconds", type=float, default=1.0)
    args = p.parse_args(argv)
    try:
        root = check_root(args.root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = root / "quality" / ("listening_queue_" + timestamp)
        report, sheet = create_queue(root, folder, args.per_meeting,
                                     args.window_seconds, args.pad_seconds)
        print("REAL AUDIO LISTENING QUEUE CREATED (UNREVIEWED)")
        for mid, row in report["meetings"].items():
            print(mid, "independent nonoverlap reference fraction",
                  row["nonoverlap_nonzero_fraction"],
                  "word-overlap fraction", row["other_speaker_overlap_fraction"],
                  "selected clips", row["selected_clip_count"])
        print("TOTAL REVIEW CLIPS", report["selected_clips"])
        print("PRIVATE CSV", sheet)
        print("PRIVATE INVENTORY", folder / "queue_inventory.json")
        print("HUMAN GOLD NOT_CREATED. RELEASE BLOCKED.")
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print("AUDIO REVIEW QUEUE NOT_READY:", type(exc).__name__,
              str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

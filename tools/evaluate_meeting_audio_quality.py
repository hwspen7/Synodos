"""Offline AMI Whisper-versus-manual-reference screening; never QA gold."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from tools.prepare_meeting_reference_corpus import ASSETS, MEETINGS, sha256_file
from tools.prepare_real_meeting_evaluation import (
    DEFAULT_ROOT, REFERENCE_SCHEMA, check_root, load_predictions,
    validate_prediction,
)

SCHEMA = "meeting-real-audio-quality-screening"
FIELDS = ("meeting", "start_s", "end_s", "priority",
          "reference_words", "predicted_words", "cross_talk_fraction",
          "screening_wer", "asr_segment_ids", "official_reference_text",
          "whisper_prediction_text", "review_status", "review_notes")
WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)


def tokenize(text):
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def edit_distance(a, b):
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, token in enumerate(b, 1):
        row = [i]
        for j, other in enumerate(a, 1):
            row.append(min(row[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (token != other)))
        prev = row
    return prev[-1]


def cross_talk_count(words):
    words = sorted(words, key=lambda x: x["start_ms"])
    touched = set()
    for i, a in enumerate(words):
        if a["end_ms"] <= a["start_ms"]:
            continue
        for j in range(i + 1, len(words)):
            b = words[j]
            if b["start_ms"] >= a["end_ms"]:
                break
            if (a["speaker"] != b["speaker"] and
                    b["end_ms"] > a["start_ms"] and b["end_ms"] > b["start_ms"]):
                touched.update((i, j))
    return len(touched)


def verify_reference(doc, meeting, audio_hash, archive_hash, duration):
    if (doc.get("schema") != REFERENCE_SCHEMA or
            doc.get("meeting") != meeting or
            doc.get("source_audio_sha256") != audio_hash or
            doc.get("annotation_archive_sha256") != archive_hash or
            doc.get("independently_reviewed_against_mixed_audio") is not False or
            doc.get("is_qa_gold") is not False or
            doc.get("is_machine_prediction") is not False):
        raise ValueError("Unverified official reference provenance")
    words = doc.get("words")
    if not isinstance(words, list) or len(words) < 100:
        raise ValueError("Incomplete reference words")
    for item in words:
        if (not isinstance(item, dict) or
                not isinstance(item.get("text"), str) or not item["text"].strip() or
                not isinstance(item.get("speaker"), str) or not item["speaker"] or
                type(item.get("start_ms")) is not int or
                type(item.get("end_ms")) is not int or
                not 0 <= item["start_ms"] <= item["end_ms"] <=
                round(duration * 1000) + 2000):
            raise ValueError("Invalid reference timestamps")
    return words


def evaluate_windows(mid, duration, reference, prediction, seconds):
    if not 10 <= seconds <= 120:
        raise ValueError("Window seconds must be between 10 and 120")
    count = math.ceil(duration / seconds)
    refs, preds = defaultdict(list), defaultdict(list)
    for word in reference:
        midpoint = (word["start_ms"] + word["end_ms"]) / 2000
        refs[min(count - 1, int(midpoint // seconds))].append(word)
    for segment in prediction:
        midpoint = (segment["start_ms"] + segment["end_ms"]) / 2000
        preds[min(count - 1, int(midpoint // seconds))].append(segment)
    rows = []
    clean_words = clean_edits = clean_windows = 0
    for k in range(count):
        rw = sorted(refs[k], key=lambda x: (x["start_ms"], x["end_ms"], x["speaker"]))
        pw = sorted(preds[k], key=lambda x: (x["start_ms"], x["end_ms"]))
        text_ref = " ".join(w["text"] for w in rw)
        text_pred = " ".join(p["text"] for p in pw)
        a, b = tokenize(text_ref), tokenize(text_pred)
        overlap = cross_talk_count(rw) / len(rw) if rw else 0
        edits = edit_distance(a, b)
        if len(a) >= 5 and overlap <= .01:
            clean_words += len(a)
            clean_edits += edits
            clean_windows += 1
        priority = ("OVERLAP_REVIEW" if overlap > .01 else
                    "MISSING_ASR" if a and not b else
                    "ASR_WITHOUT_REFERENCE" if b and not a else
                    "HIGH_DISAGREEMENT" if a and edits / len(a) > .4 else
                    "SPOT_CHECK")
        rows.append(dict(
            meeting=mid, start_s=round(k * seconds, 2),
            end_s=round(min(duration, (k + 1) * seconds), 2),
            priority=priority, reference_words=len(a), predicted_words=len(b),
            cross_talk_fraction=round(overlap, 4),
            screening_wer=round(edits / len(a), 4) if a else None,
            asr_segment_ids=",".join(str(p["id"]) for p in pw),
            official_reference_text=text_ref,
            whisper_prediction_text=text_pred,
            review_status="UNREVIEWED", review_notes=""))
    return dict(
        window_count=count, low_overlap_reference_words=clean_words,
        low_overlap_word_edits=clean_edits,
        low_overlap_window_count=clean_windows,
        low_overlap_screening_wer=round(clean_edits / clean_words, 5)
            if clean_words else None,
        windows_to_review=sum(r["priority"] != "SPOT_CHECK" for r in rows),
    ), rows


def evaluate(root, seconds):
    archive = root / "annotations/ami_public_manual_1.6.2.zip"
    archive_sha = dict(ASSETS)["annotations/ami_public_manual_1.6.2.zip"]
    if not archive.is_file() or sha256_file(archive) != archive_sha:
        raise ValueError("Official annotation archive absent or altered")
    predictions = load_predictions(root)
    if {x[0] for x in predictions} != set(MEETINGS):
        raise ValueError("Incomplete AMI meeting cohort")
    stats, reviews = {}, []
    for mid, audio, duration, segments, doc in predictions:
        audio_sha = sha256_file(audio)
        validate_prediction(doc, mid, audio_sha, duration)
        ref_path = root / "reference" / (mid + "_official_words.json")
        if not ref_path.is_file():
            raise FileNotFoundError("Missing official words; run reference first: " + str(ref_path))
        words = verify_reference(json.loads(ref_path.read_text(encoding="utf-8")),
                                 mid, audio_sha, archive_sha, duration)
        summary, rows = evaluate_windows(mid, duration, words, segments, seconds)
        stats[mid] = dict(
            asr_segment_count=len(segments), official_word_count=len(words),
            source_audio_sha256=audio_sha,
            asr_prediction_sha256=sha256_file(
                root / "predictions" / (mid + "_whisper.json")),
            official_reference_sha256=sha256_file(ref_path),
            **summary,
        )
        reviews.extend(rows)
    total_words = sum(s["low_overlap_reference_words"] for s in stats.values())
    total_edits = sum(s["low_overlap_word_edits"] for s in stats.values())
    report = dict(
        schema=SCHEMA, meeting_count=len(stats),
        asr_segment_count=sum(s["asr_segment_count"] for s in stats.values()),
        window_count=len(reviews), window_seconds=seconds,
        low_overlap_screening_wer=round(total_edits / total_words, 5)
            if total_words else None,
        per_meeting=stats, qa_gold_ready=False, product_release_pass=False,
        independent_audio_review_performed=False,
        limitations=[
            "Manual reference contains overlapping speakers not always audible in the mono mix.",
            "Time buckets use utterance midpoints; boundary mismatch can inflate error rates.",
            "Low-overlap WER is a screening signal, not final WER or a release gate.",
        ],
    )
    return report, reviews


def save_report(folder, report, rows):
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary = folder / ("meeting_audio_quality_" + stamp + ".json")
    sheet = folder / ("meeting_audio_review_" + stamp + ".csv")
    if summary.exists() or sheet.exists():
        raise FileExistsError("Review artifacts already exist; refusing overwrite")
    with sheet.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with summary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return summary, sheet


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--window-seconds", type=int, default=30)
    args = p.parse_args(argv)
    try:
        root = check_root(args.root)
        report, rows = evaluate(root, args.window_seconds)
        summary, csv_file = save_report(root / "quality", report, rows)
        print("REAL MEETING AUDIO QUALITY SCREENING COMPLETE")
        for mid, item in report["per_meeting"].items():
            print(mid, "ASR segments", item["asr_segment_count"],
                  "low_overlap_screening_wer", item["low_overlap_screening_wer"],
                  "review_windows", item["windows_to_review"])
        print("TOTAL ASR SEGMENTS", report["asr_segment_count"])
        print("SUMMARY", summary)
        print("REVIEW CSV", csv_file)
        print("QA GOLD NOT_READY; RELEASE BLOCKED")
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print("AUDIO QUALITY SCREENING NOT_READY:", type(exc).__name__,
              str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

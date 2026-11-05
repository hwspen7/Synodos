"""Prepare independent real AMI audio, ASR hypotheses and private test records.

Official NXT references are NEVER represented as machine ASR predictions or as
independently audio-reviewed QA gold. PostgreSQL writes require explicit apply.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import wave
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from tools.prepare_meeting_reference_corpus import ASSETS, MEETINGS, sha256_file

DEFAULT_ROOT = Path(r"E:\AI-Cache\MeetingReference\AMI")
REFERENCE_SCHEMA = "meeting-official-reference-words"
PREDICTION_SCHEMA = "meeting-real-audio-asr"
MEETING_PREFIX = "ami_eval_"
MAX_XML_BYTES = 4 * 1024 * 1024


def evaluation_id(meeting: str) -> str:
    if meeting not in MEETINGS:
        raise ValueError("Unrecognized reference meeting")
    return MEETING_PREFIX + meeting.lower()


def check_root(root: Path) -> Path:
    root = root.resolve()
    if os.name == "nt" and root.drive.casefold() != "e:":
        raise ValueError("Reference data must remain on E drive")
    return root


def source_audio(root: Path, meeting: str) -> tuple[Path, float]:
    if meeting not in MEETINGS:
        raise ValueError("Unsupported AMI meeting")
    relative = f"sdm/{meeting}.Mix-Headset.wav"
    path = root / relative
    if not path.is_file() or sha256_file(path) != dict(ASSETS)[relative]:
        raise ValueError("Missing or unverified source WAV: " + meeting)
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Unexpected PCM format: " + meeting)
        duration = reader.getnframes() / 16000
    if duration < 300:
        raise ValueError("Source recording is too short")
    return path, duration


def write_private_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def parse_reference_archive(archive: Path, meeting: str, duration: float) -> tuple[list[dict], dict]:
    """Read original speaker-word annotations, not predictions or QA gold."""
    words = []
    source_files = {}
    excluded = 0
    with ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            match = re.fullmatch(re.escape(meeting) + r"\.([A-Z])\.words\.xml", name)
            if not match:
                continue
            if info.is_dir() or info.file_size > MAX_XML_BYTES:
                raise ValueError("Invalid word XML size")
            if name in source_files:
                raise ValueError("Duplicate speaker XML: " + name)
            raw = z.read(info)
            import hashlib
            source_files[name] = hashlib.sha256(raw).hexdigest()
            root = ET.fromstring(raw)
            for element in root.iter():
                if str(element.tag).split("}")[-1] != "w":
                    continue
                text = (element.text or "").strip()
                if not text or (text.startswith("<") and text.endswith(">")):
                    continue
                try:
                    start = float(element.attrib["starttime"])
                    end = float(element.attrib["endtime"])
                except (KeyError, ValueError, TypeError):
                    excluded += 1
                    continue
                if not math.isfinite(start) or not math.isfinite(end) or end < start:
                    excluded += 1
                    continue
                if not 0 <= start <= end <= duration + 2:
                    excluded += 1
                    continue
                words.append({"speaker": match.group(1), "start_ms": round(start * 1000),
                              "end_ms": round(end * 1000), "text": text})
    if len(source_files) < 2 or len(words) < 100:
        raise ValueError("Insufficient independent reference annotations for " + meeting)
    words.sort(key=lambda w: (w["start_ms"], w["end_ms"], w["speaker"]))
    return words, {"xml_sha256": source_files, "excluded_lexical_entries": excluded,
                   "overlap_note": "Reference annotations include overlapping speakers and may not all be audible from the mix."}


def prepare_reference(root: Path) -> None:
    archive = root / "annotations/ami_public_manual_1.6.2.zip"
    if not archive.is_file() or sha256_file(archive) != dict(ASSETS)["annotations/ami_public_manual_1.6.2.zip"]:
        raise ValueError("Missing or unverified manual annotation archive")
    for meeting in MEETINGS:
        audio, duration = source_audio(root, meeting)
        destination = root / "reference" / (meeting + "_official_words.json")
        if destination.exists():
            saved = json.loads(destination.read_text(encoding="utf-8"))
            if (saved.get("schema") != REFERENCE_SCHEMA or saved.get("meeting") != meeting or
                    saved.get("source_audio_sha256") != sha256_file(audio) or
                    saved.get("annotation_archive_sha256") != sha256_file(archive)):
                raise ValueError("Existing reference does not match the source. No overwrite: " + str(destination))
            print("REFERENCE VERIFIED", meeting, len(saved["words"]), "words")
            continue
        words, details = parse_reference_archive(archive, meeting, duration)
        write_private_json(destination, {
            "schema": REFERENCE_SCHEMA, "meeting": meeting,
            "source": "AMI Consortium manual NXT word annotations (CC BY 4.0)",
            "source_audio_sha256": sha256_file(audio),
            "annotation_archive_sha256": sha256_file(archive),
            "independently_reviewed_against_mixed_audio": False,
            "is_machine_prediction": False, "is_qa_gold": False,
            "duration_seconds": round(duration, 3), "words": words, **details,
        })
        print("REFERENCE PREPARED", meeting, len(words), "words", destination)


def validate_prediction(doc: dict, meeting: str, audio_sha256: str, duration_s: float) -> list[dict]:
    if (doc.get("schema") != PREDICTION_SCHEMA or doc.get("meeting") != meeting or
            doc.get("source_audio_sha256") != audio_sha256 or
            doc.get("reference_used_for_inference") is not False or
            doc.get("independently_audio_reviewed") is not False or
            doc.get("pipeline") != "speech.whisper.WhisperEngine"):
        raise ValueError("Invalid prediction provenance for " + meeting)
    segments = doc.get("segments")
    if not isinstance(segments, list) or len(segments) < 10:
        raise ValueError("Insufficient actual ASR utterances for " + meeting)
    previous_start = -1
    for i, row in enumerate(segments):
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] != i:
            raise ValueError("Invalid ASR segment identifier")
        start, end = row.get("start_ms"), row.get("end_ms")
        if (type(start) is not int or type(end) is not int or
                not 0 <= start < end <= round(duration_s * 1000) + 1 or
                start < previous_start or not isinstance(row.get("text"), str) or
                not row["text"].strip()):
            raise ValueError("Invalid ASR text or absolute timestamp")
        previous_start = start
    return segments


def load_predictions(root: Path) -> list[tuple[str, Path, float, list[dict], dict]]:
    reports = []
    for meeting in MEETINGS:
        audio, duration = source_audio(root, meeting)
        path = root / "predictions" / (meeting + "_whisper.json")
        if not path.is_file():
            raise FileNotFoundError("Real ASR prediction missing. Run transcribe first: " + str(path))
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = validate_prediction(doc, meeting, sha256_file(audio), duration)
        reports.append((meeting, audio, duration, rows, doc))
    models = {d[4]["model"] for d in reports}
    if len(models) != 1:
        raise ValueError("Model identity changed between meeting recordings")
    return reports


def transcribe(root: Path, model: str, device: str, compute_type: str, chunk_seconds: int) -> None:
    if chunk_seconds < 15 or chunk_seconds > 120:
        raise ValueError("Chunk duration must be between 15 and 120 seconds")
    # Set the local cache BEFORE importing Whisper or downloading model weights.
    os.environ.setdefault("HF_HOME", str(root.parents[1] / "HuggingFace"))
    import numpy as np
    from speech.whisper import WhisperEngine
    engine = WhisperEngine(model, device=device, compute_type=compute_type)
    padding_frames = 16000 * 2
    step = 16000 * chunk_seconds
    for meeting in MEETINGS:
        audio, duration = source_audio(root, meeting)
        destination = root / "predictions" / (meeting + "_whisper.json")
        if destination.exists():
            doc = json.loads(destination.read_text(encoding="utf-8"))
            validate_prediction(doc, meeting, sha256_file(audio), duration)
            if doc.get("model") != model:
                raise ValueError("Existing prediction uses a different ASR model: " + meeting)
            print("ASR VERIFIED EXISTING", meeting)
            continue
        output = []
        chunk_count = 0
        with wave.open(str(audio), "rb") as wav:
            total = wav.getnframes()
            for core_start in range(0, total, step):
                core_end = min(total, core_start + step)
                read_start = max(0, core_start - padding_frames)
                read_end = min(total, core_end + padding_frames)
                wav.setpos(read_start)
                pcm = wav.readframes(read_end - read_start)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                recognized = engine.recognize(samples, language="en")
                chunk_count += 1
                for u in recognized.utterances:
                    if not u.text.strip():
                        continue
                    start_ms = round(read_start * 1000 / 16000) + int(u.start_ms)
                    end_ms = round(read_start * 1000 / 16000) + int(u.end_ms)
                    center_frame = round((start_ms + end_ms) * 16000 / 2000)
                    if not core_start <= center_frame < core_end:
                        continue
                    if start_ms < 0 or end_ms > round(duration * 1000) + 1 or end_ms <= start_ms:
                        raise ValueError("ASR returned invalid utterance timestamps")
                    output.append({"start_ms": start_ms, "end_ms": end_ms,
                                   "text": u.text.strip(), "language": u.language or "en",
                                   "timestamp_quality": u.timestamp_quality,
                                   "recognition_origin": u.recognition_origin})
                print("ASR", meeting, "chunk", chunk_count, "of", math.ceil(total / step), flush=True)
        output.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
        for idx, row in enumerate(output):
            row["id"] = idx
        doc = {"schema": PREDICTION_SCHEMA, "meeting": meeting, "model": model,
               "device": device, "compute_type": compute_type,
               "pipeline": "speech.whisper.WhisperEngine",
               "source_audio_sha256": sha256_file(audio),
               "source_audio_path": str(audio), "duration_seconds": round(duration, 3),
               "reference_used_for_inference": False,
               "independently_audio_reviewed": False,
               "speaker_identity_verified": False,
               "time_boundary_note": "Two-second context padding, center-of-utterance ownership; inspect boundary omissions and duplication.",
               "chunk_seconds": chunk_seconds, "chunks_processed": chunk_count,
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "segments": output}
        validate_prediction(doc, meeting, sha256_file(audio), duration)
        write_private_json(destination, doc)
        print("REAL AUDIO ASR PREPARED", meeting, len(output), "utterances", destination)


def make_state(meeting: str, duration: float, segments: list[dict], doc: dict) -> tuple[dict, dict]:
    """Build source-attributed ASR-only records without fabricated speakers."""
    mid = evaluation_id(meeting)
    now = datetime.now(timezone.utc).isoformat()
    state_segments = [{"id": r["id"], "start_sample": round(r["start_ms"] * 16),
                       "end_sample": round(r["end_ms"] * 16), "text": r["text"],
                       "language": r.get("language") or "en",
                       "timestamp_quality": r.get("timestamp_quality") or "segment",
                       "recognition_origin": r.get("recognition_origin") or "primary",
                       "speaker_id": -1, "cluster": -1,
                       "speaker_name": "Unidentified Speaker", "speaker_status": "unknown"}
                      for r in segments]
    state = {"id": mid, "title": f"AMI {meeting} - Whisper ASR (unreviewed)",
             "sample_rate": 16000, "segments": state_segments, "speaker_map": {},
             "reference_source": "AMI Meeting Corpus, CC BY 4.0",
             "audio_sha256": doc["source_audio_sha256"],
             "asr_model": doc["model"], "human_audio_reviewed": False,
             "diarization_evaluated": False}
    meta = {"id": mid, "title": state["title"], "created_at": now,
            "updated_at": now, "sample_rate": 16000,
            "segments": len(state_segments), "duration_ms": round(duration * 1000),
            "speakers": 0, "corpus": "AMI", "source_meeting_id": meeting,
            "human_audio_reviewed": False}
    return meta, state


def inspect_db_name(dsn: str) -> None:
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        if conn.info.dbname != "meeting_test":
            raise RuntimeError("Refusing to access a database other than meeting_test")
        conn.execute("SELECT 1 FROM meeting_records LIMIT 1")


def import_predictions(root: Path, apply: bool) -> None:
    reports = load_predictions(root)
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("DATABASE_URL must point to meeting_test")
    inspect_db_name(dsn)
    print("IMPORT PLAN: 3 separate real audio ASR records into meeting_test")
    for meeting, audio, duration, rows, doc in reports:
        print(evaluation_id(meeting), "ASR segments", len(rows), "duration_s", round(duration, 2),
              "audio_sha256", doc["source_audio_sha256"])
    if not apply:
        print("DRY RUN ONLY: pass --apply to insert records. Existing records are never overwritten.")
        return
    from psycopg.types.json import Jsonb
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        if conn.info.dbname != "meeting_test":
            raise RuntimeError("Refusing to write to a database other than meeting_test")
        with conn.transaction():
            ids = [evaluation_id(m) for m, *_ in reports]
            existing = conn.execute("SELECT id FROM meeting_records WHERE id = ANY(%s)", (ids,)).fetchall()
            if existing:
                raise ValueError("Import collision. Existing meetings will not be modified: " + ", ".join(x[0] for x in existing))
            for meeting, audio, duration, rows, doc in reports:
                meta, state = make_state(meeting, duration, rows, doc)
                mid = state["id"]
                conn.execute("INSERT INTO meeting_records(id,meta,state,revision) VALUES (%s,%s,%s,1)",
                             (mid, Jsonb(meta), Jsonb(state)))
                # Psycopg 3 provides executemany on Cursor, not Connection.
                # Both writes remain in the surrounding transaction.
                with conn.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO meeting_segments(meeting_id,segment_id,start_sample,end_sample,speaker_id,content) "
                        "VALUES (%s,%s,%s,%s,%s,%s)",
                        [(mid, s["id"], s["start_sample"], s["end_sample"], -1, s["text"])
                         for s in state["segments"]])
    print("IMPORTED REAL ASR: 3 meetings into meeting_test (never human-reviewed gold)")


def index_predictions(root: Path) -> None:
    reports = load_predictions(root)
    from tools.meeting_knowledge_synthetic import guard_environment
    guard_environment()
    from intelligence.config import load_config
    from intelligence.model_gateway import ModelGateway
    from intelligence import store
    from intelligence.worker import process
    cfg = load_config()
    if not cfg.enabled or cfg.provider != "ollama" or not cfg.enable_embeddings:
        raise RuntimeError("Set MEETING_AI_ENABLED=1 and MEETING_AI_EMBEDDINGS=1 for hybrid indexing")
    gateway = ModelGateway(cfg)
    if not gateway.embed("real meeting reference indexing preflight"):
        raise RuntimeError("Embedding model unavailable")
    for meeting, audio, duration, rows, doc in reports:
        mid = evaluation_id(meeting)
        revision, recorded = store.meeting_segments(mid)
        if len(recorded) != len(rows) or any(a["text"] != b["text"] for a, b in zip(recorded, rows)):
            raise ValueError("Stored ASR does not match this immutable prediction: " + meeting)
        print("INDEX START", mid, "segments", len(rows), flush=True)
        result = process({"kind": "index_meeting", "meeting_id": mid, "payload": {}, "id": "manual"}, gateway)
        if result.get("source_revision") != revision:
            raise RuntimeError("Index version mismatched its source")
        print("INDEX READY", mid, "chunks", result.get("indexed_chunks"), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("reference", "transcribe", "import", "index"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--chunk-seconds", type=int, default=30)
    p.add_argument("--apply", action="store_true", help="Required before PostgreSQL changes")
    args = p.parse_args(argv)
    try:
        root = check_root(args.root)
        if args.action == "reference":
            prepare_reference(root)
        elif args.action == "transcribe":
            transcribe(root, args.model, args.device, args.compute_type, args.chunk_seconds)
        elif args.action == "import":
            import_predictions(root, args.apply)
        else:
            index_predictions(root)
        return 0
    except Exception as exc:
        print("REAL MEETING PREPARATION NOT_READY:", type(exc).__name__ + ":", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

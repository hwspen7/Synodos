# Real meeting audio and knowledge evaluation

Run the same Whisper ASR adapter used by Meeting against three **recorded** AMI meetings, preserve their separate official NXT reference transcription and then explicitly import **unreviewed Whisper predictions only** into the isolated PostgreSQL `meeting_test` database. This does not assert real-meeting RAG release acceptance or speech diarization quality.

**Corpora and attribution.** AMI Consortium, CC BY 4.0. Audio and official NXT transcript archive obtained from the existing checked mirror at `E:\AI-Cache\MeetingReference\AMI`. These recordings are English. They do not replace multilingual recognition or speaker-diarization evaluations.

## Step 1: extract official reference (read-only, offline)

```powershell
cd E:\Meeting-Fixed10-Test
& .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_real_meeting_evaluation reference
```

The private `reference/<meeting>_official_words.json` files contain time-coded words attributed to the original annotation speaker and explicit `independently_reviewed_against_mixed_audio: false`. The transcription is **not** fed to Whisper and is not stored as ASR output. Because separate speakers can overlap in the mixed audio, global WER calculated by simply concatenating all reference word streams would be misleading.

## Step 2: run genuine audio ASR (GPU, no database writes)

```powershell
& .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_real_meeting_evaluation transcribe `
    --model large-v3 --device cuda --compute-type float16
```

The tool calls `speech.asr.whisper.WhisperEngine` with 30-second core windows and two seconds of extra context at either end, retains recognized utterances based on their midpoint, restores absolute timestamps, and saves separate private `predictions/<meeting>_whisper.json` files. Verify boundary duplicates/omissions from the source recordings before treating timing or lexical metrics as final. It reuses an existing, correctly sourced prediction and refuses overwrite. Failed runs do not create a passing prediction. The source WAV hash is checked before processing. If unset, `HF_HOME` is set to an E-drive cache before model load. An existing `HF_HOME` value is preserved, so ensure that any pre-existing cache configuration also points to E drive.

## Step 3: inspect then explicitly import (PostgreSQL)

```powershell
& .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_real_meeting_evaluation import
& .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_real_meeting_evaluation import --apply
```

`DATABASE_URL` must point to `meeting_test`. The importer validates **all three** complete prediction files and every corresponding original WAV before any insertion. It fails on existing IDs and inserts all three records in one PostgreSQL transaction. Imported IDs are `ami_eval_en2002a`, `ami_eval_es2004a`, and `ami_eval_is1009a`. The source reference transcription is never copied into these records. Speaker identities are `Unidentified Speaker`, not fabricated. The import does **not** run against the live `meeting` database or alter existing source recordings.

## Step 4: index actual ASR transcripts for real Lexical and Hybrid search

```powershell
$env:MEETING_STORAGE_BACKEND = 'postgres'
$env:MEETING_AI_ENABLED = '1'
$env:MEETING_AI_PROVIDER = 'ollama'
$env:MEETING_AI_EMBEDDINGS = '1'
$env:MEETING_AI_EMBED_MODEL = 'qwen3-embedding:0.6b'
$env:MEETING_AI_ENDPOINT = 'http://127.0.0.1:11434'
& .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_real_meeting_evaluation index
```

Use the endpoint where the verified embedder is actually running. Indexing uses the existing Meeting knowledge pipeline and writes only to the three isolated `meeting_test` source IDs. Run it only when no live Meeting session is active. The index may be rebuilt if the first attempt is interrupted.

## Step 5: independent review, then the release gate

The normal export helper produces an **unreviewed** CSV and draft QA gold manifest:

```powershell
& .\.venv\Scripts\python.exe -m tools.diagnostics.meeting_audio_review_export `
    --meeting-id ami_eval_en2002a --meeting-id ami_eval_es2004a --meeting-id ami_eval_is1009a `
    --audio-path 'E:\AI-Cache\MeetingReference\AMI\sdm\EN2002a.Mix-Headset.wav' `
    --audio-path 'E:\AI-Cache\MeetingReference\AMI\sdm\ES2004a.Mix-Headset.wav' `
    --audio-path 'E:\AI-Cache\MeetingReference\AMI\sdm\IS1009a.Mix-Headset.wav' `
    --csv 'E:\AI-Cache\MeetingResults\meeting_audio_review.csv' `
    --manifest 'E:\AI-Cache\MeetingResults\meeting_audio_review_gold.json'
```

Listen to the corresponding original audio, consult the **separate** official word annotations and verify at least 20 positive question-answer pairs plus six negatives, with contradictions and cross-meeting isolation. Fill accurate segment IDs and absolute audio intervals. Only independently verified facts can set the review fields to true. Do not commit personal comments, audio, predictions, official annotations or draft gold labels to public GitHub. Finally run `tools/Invoke-Meeting-Quality-Checks.ps1`. It will remain `NOT_ACCEPTED` until the required human review and independent audio/identity release gates exist.

## Failure handling

The tool never deletes a recording, truncates existing files, replaces a meeting or manufactures an independent gold label. A partially completed ASR run has no output file for that meeting, so the next invocation retries only missing files. If a predicted audio boundary is incorrect, create a separately named corrected human-reviewed artifact rather than silently rewriting the model prediction.

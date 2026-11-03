# Fixed150: reproducible multilingual and meeting-speech evaluation

**Scope:** 100 original single-speaker recordings (56 MInDS-14 + 44 FLEURS),
plus **50 real or source-grounded multi-speaker recordings** (25 two-person
AMI natural meeting excerpts + 25 LAION three-person multilingual overlaps).
40 optional deterministic acoustic perturbations are **additional correlated
stress variants**, never counted as 40 new people or independent recordings.

This is a collection and evaluation **pipeline**, not a claim that the project's
author has downloaded or inferred on these 150 WAVs. Measured scores come
only from the user's actual GPU inference and meeting backend.

## Source matrix

| Count | Corpus | Speech type | Linguistic / acoustic evidence |
| ---: | --- | --- | --- |
| 56 | PolyAI MInDS-14 | Recorded banking intent requests | 14 locale codes, including UK/AU/US English; natural recording variability, no asserted noise label |
| 44 | Google FLEURS | Read speech | 11 additional languages, 4 utterances each, multiple scripts |
| 25 | Trelis/AMI 2-speaker | As-recorded real English meetings | Interruptions, short acknowledgements, overlap, actual turn timestamps |
| 25 | LAION small overlapping bench | 3 voices mixed from independent multilingual speech | 3 overlapping languages/speakers, reference transcript and placement intervals |

Total **23 distinct single-speaker language codes**:
cs, de, en, es, fr, it, ko, nl, pl, pt, ru, zh, ar, hi, ja, th, vi,
tr, id, uk, bn, fa, sw. Original locales also distinguish English accents.
Multi-language overlaps add speaker combinations from de, fr, es, it, nl and pt.
This is **not** proof of quality for all 102 FLEURS languages.

**Dataset revision caveat (verified for google/fleurs@70bb2e8):**
FLEURS validation Parquet shards are under
`parquet-data/<locale>/validation-00000-of-00001.parquet`, **not**
`<locale>/validation-00000-of-00001.parquet`.
The preparation program now resolves the real path from the pinned Hugging
Face repository tree before starting the large download. A known-path failure
blocks preparation without discarding already-verified WAVs and the
`progress.json` checkpoints. Do not delete `benchmark/fixed150` to fix a
404.

The 150-case manifest is source-pinned and SHA-256 verified. Case selection is
deterministic: MInDS row indices 19/43/77/101 from 14 locales, FLEURS validation
indices 19/43/77/101 from 11 locales, AMI source indices 0/2/.../48, and
LAION source indices 0/4/.../96. All IDs are independent within the set.
A progress file is written after **each completed WAV** so interrupted
downloads can resume without overwriting verified recordings. The final lock is
issued **only** when 150 valid unique WAV files and both manifests exist.

### Source attributions and licenses

- https://huggingface.co/datasets/PolyAI/minds14  — CC-BY-4.0
- https://huggingface.co/datasets/google/fleurs — CC-BY-4.0
- https://huggingface.co/datasets/Trelis/ami-2speaker-test — CC-BY-4.0,
  derived from AMI (Carletta et al.)
- https://huggingface.co/datasets/laion/small-overlapping-speech-bench —
  CC-BY-4.0, derived from Multilingual LibriSpeech (Pratap et al.)

The AMI extract is 37.1 MB for 50 source clips; this pipeline selects 25.
The LAION source repository holds 100 audio clips; this pipeline selects 25.
FLEURS validation Parquet files can exceed **200 MB per language**. Plan for
several GB of transient source downloads and extra E-drive space. Downloads
may run for a long time; this script intentionally avoids automatically
cleaning immutable HF source caches, which would break resume capability.

## Windows PowerShell: put ALL caches and scratch files on E

Start from your existing feature worktree:

    Set-Location "E:\Meeting-Fixed10-Test"
    git fetch origin feature/multilingual-fixed10-ui-refresh
    git switch --detach origin/feature/multilingual-fixed10-ui-refresh

Use a **new PowerShell terminal** for the following session configuration.
The following paths are for the user's actual E: drive:

    New-Item -ItemType Directory -Force "E:\AI-Cache\uv", "E:\AI-Cache\temp", "E:\AI-Cache\huggingface" | Out-Null
    $env:UV_CACHE_DIR = "E:\AI-Cache\uv"
    $env:HF_HOME = "E:\AI-Cache\huggingface"
    $env:TEMP = "E:\AI-Cache\temp"
    $env:TMP = "E:\AI-Cache\temp"
    $env:TMPDIR = "E:\AI-Cache\temp"

Install prerequisites in the **E: worktree** virtual environment:

    python -m uv pip install --python "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe" huggingface_hub pyarrow soundfile scipy numpy plotly websockets faster-whisper

The Windows warning about Hugging Face symlinks is informational; enable
Developer Mode if desired. A missing HF_TOKEN may lower API rate limits but
does not invalidate accessible source content. Do not echo a private token
into a screenshot or log.

### 1. Prepare and verify 150 original audio files

    .\.venv\Scripts\python.exe tools\fixed150_prepare.py
    .\.venv\Scripts\python.exe tools\fixed150_prepare.py --verify-only

If network, disk, or a single source fails during preparation, run the first
command again: successfully prepared WAVs are SHA-checked and skipped.
The script does **not** claim a finished suite until the lock and 100+50
manifests are created and verified. Progress is written to
benchmark/fixed150/progress.json; incomplete files are not "verified."

Expected on genuine success:

    LOCK VERIFIED: 150 unique WAV files, 100 singles, 50 multispeakers

### 2. Baseline on 100 single-speaker recordings

    .\.venv\Scripts\python.exe tools\phase07_benchmark.py --manifest benchmark\fixed150\manifest_single.jsonl --output benchmark\fixed150\single_asr.json --infer --engines whisper --whisper-model large-v3 --device cuda --compute-type float16 --warmup-whisper

The new flag preloads Whisper weights before timing and writes
whisper_model_load_ms separately in the report. First-time CUDA VAD or kernels
can still contribute some initialization cost; for reproducible steady-state
measurements, repeat the inference on a warm process and report both cold and
warm results. GPU execution must be real, not an offline scoring-only run.
Each sample is language-auto, not pre-forced to the reference language.

### 3. Real-time audio replay with 50 multi-speaker recordings

Start the full service in the first terminal using the project's normal
configured backend dependencies and GPU environment:

    .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

In a **second** terminal, set the E-drive environment variables above and run:

    .\.venv\Scripts\python.exe tools\fixed150_replay.py --scope multi --limit 3
    .\.venv\Scripts\python.exe tools\fixed150_replay.py --scope multi

The first command checks 3 clips before committing to the 50-case full
replay. The second resumes them and continues. The replay verifies checksum
and exact PCM-sample acknowledgement and records transcript segments.
Each completed case is persisted in benchmark/fixed150/replay_results.json
immediately. It must be run against the **actual meeting WebSocket backend**.
By default audio is paced at 1.0x real time. No fake transcript is produced.
To exercise all 150 through the live pipeline, pass --scope all.

### 4. Optional 40 synthetic robustness conditions

    .\.venv\Scripts\python.exe tools\fixed150_stress.py
    .\.venv\Scripts\python.exe tools\phase07_benchmark.py --manifest benchmark\fixed150\manifest_stress.jsonl --output benchmark\fixed150\stress_asr.json --infer --engines whisper --whisper-model large-v3 --device cuda --compute-type float16 --warmup-whisper

The 40 variants are 8 original WAVs x 5 *artificial* signal changes:
15 dB SNR noise, 5 dB SNR noise, synthetic room echo, 300–3400 Hz telephone
bandwidth, and reduced volume. They are **paired sensitivity probes**, not
40 new independent recordings. Record the difference from each unmodified
base, not a misleading larger independent sample size.

### 5. Evidence dashboard

    .\.venv\Scripts\python.exe tools\fixed150_dashboard.py
    Start-Process .\benchmark\fixed150\dashboard.html

Produces dashboard.html and scorecard.json, with six Plotly panels for
per-language CER, cross-script token error, original source, time latency,
speaker-attributed recovery vs pooled content recall and diarization error.

Multi-speaker content recall is scored *after* speaker permutation alignment
using the diarization evaluation mapping. A pooled transcript that mentions
all words but assigns them to the wrong speaker must **not** get full
speaker-attributed credit.

The tool reports missing sources, failed inference, unacknowledged audio,
invalid timestamps and missing predictions as BLOCKED, not zero error.
AMI source annotations contain timestamped utterance segments. LAION
overlap timing reflects utterance placement, not fine phonetic boundaries.
Their DERs should be reported separately.

## Coverage and limitations

This is considerably more comprehensive than Fixed10, but cannot alone
certify a Zoom-level meeting transcription product. Still required as
independent, consented evaluation tracks:

- Real microphone recordings at varied distances, laptop/USB/Bluetooth
  headsets, codec packet loss, room geometry, HVAC, traffic and keyboard
  background noise. The synthetic suite is not a substitute.
- Four or more simultaneous natural speakers, speaker re-entry, unknown
  speaker and name enrollment, code-switching within the same sentence,
  severe accents, whispering, shouting and laughter.
- Real uninterrupted 60-minute and 120-minute meetings with session
  reconnection, forced tab/OS interruption, text ordering and no dropped
  audio. Report sample conservation and p50/p95/p99 *display* delay.
- Speaker-count accuracy, permutation-invariant DER, per-speaker content
  recall, transcript edits, duplications, hallucination-on-silence and tail
  truncation. Separate script-normalized CER from WER and mixed-token error.
- Held-out evaluation cases not repeatedly tuned against this fixed suite.

A release-quality score is blocked until the tests above, not merely
the 100+50 cases, have empirical data. Never represent the preparation script,
case allocation or dashboard itself as a completed GPU benchmark.

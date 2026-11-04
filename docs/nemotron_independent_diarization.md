# Independent Nemotron 3 diarization and frozen AMI acceptance

**Status: opt-in implementation, pending reference-backed model acceptance.**

The existing Whisper large-v3/CAM++ route stays the default. No model
accuracy is claimed until the original AMI WAV and RTTM are tested locally.
The goal is standard, zero-collar overlap-inclusive DER under 10% on held-out
audio, **not** a guarantee that this checkpoint will pass this exact corpus.

## Architecture

PCM frames admitted by the WebSocket are copied into TWO independent paths.

1. Whisper continues speech recognition and emits timestamped utterances.
2. A bounded background worker runs the **same continuous** Nemotron 3
   diarization stream using Transformers' official low-latency chunking API.
   Each chunk receives the prior `speaker_cache`. No speaker labels are
   reset at 120-second boundaries, and the waveform is not truncated when
   Whisper commits an utterance or reclaims its audio buffer.
3. Each caption may receive the primary Nemotron channel only when temporal
   activity, channel margin and frame coverage exceed conservative thresholds.
   These are model activity scores, NOT calibrated identity probabilities.
   No words are fabricated. Manual speaker edits remain authoritative.
4. Raw parallel speaker turns (including overlaps), source frame-count,
   cache-continuity declaration and failure state are emitted in the final
   session message. The existing anonymous transcript-derived RTTM is still
   emitted separately, so they cannot be mistaken for one another.
5. An independent AMI scorer uses the entire 240-second audio span and the
   existing pinned RTTM. A separate solo-only metric excludes overlap, and
   is always labeled **not standard DER**.

On reconnect, the old AOSC tensor cannot legitimately identify a new
anonymous session. Previously saved Nemotron labels are retained, and new
channel IDs occupy a separate namespace rather than silently reusing them.
This does NOT verify cross-reconnect identity.

If the model cannot load or its input queue is exhausted, Nemotron is
reported incomplete and CAM++ remains available. An incomplete independent
run cannot be counted as a valid quality result.

## Environment and dependency setup

Use the verified interpreter from your actual project environment.
These are PowerShell commands, not `uv` commands.

```powershell
cd E:\Meeting-Fixed10-Test
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install -r requirements-speaker-nemotron.txt
& $PY -m unittest tests.quality.test_nemotron_live_stream -v
& $PY -m unittest tests.quality.test_ami_nemotron_live_score -v
& $PY -m unittest discover -s tests -t . -v
```

The optional requirements require compatible PyTorch + CUDA runtime.
Install a suitable CUDA-enabled PyTorch build separately according to the
installed NVIDIA driver. Transformers must be >=5.18. Keep a pinned backup
of the working environment if package resolution changes the ASR stack.
Downloading model weights requires network access to Hugging Face. The
model may not support every Windows/driver/PyTorch combination.

## First terminal: launch real live server

Restart the backend after setting variables:

```powershell
cd E:\Meeting-Fixed10-Test
$env:MEETING_INDEPENDENT_DIARIZATION = "nemotron3"
$env:MEETING_NEMOTRON_STREAM_MODE = "low_latency"
$env:MEETING_NEMOTRON_DEVICE = "cuda"
$env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
$env:MEETING_ASR_TRACE_STAGES = "1"
$env:MEETING_ASR_QUALITY_CANDIDATE = "0"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "0"
$env:MEETING_WHISPER_GAP_RECOVERY = "0"
$env:MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT = "0"
$env:MEETING_ASR_COVERAGE_RETENTION = "0"
$env:MEETING_WHISPER_DUAL_VAD = "0"
& $PY -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Do not enable the previously regressed combined ASR/speaker candidates in
this first trial. Nemotron must be evaluated independently of those changes.
A separate model failure must not interrupt existing Whisper subtitles.

## Second terminal: paired real PCM replay and exact scoring

Use the verified full Python path if `.venv` is not the actual environment.

```powershell
cd E:\Meeting-Fixed10-Test
$PY = ".\.venv\Scripts\python.exe"
$root = "benchmark/fixed195/ami_shadow10"
$audio = "$root/audio/ami-TS3003b-1500-1740.wav"
$baseline = "$root/audit_TS3003b_v6_per_speaker.json"
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$replay = "$root/replay_TS3003b_nemotron_$tag.json"
$independent = "$root/audit_TS3003b_nemotron_$tag.json"
$gaps = "$root/audit_TS3003b_asr_stages_$tag.json"
& $PY -m tools.live_caption_ws_replay --audio $audio --loops 1 --speed 1.0 --final-timeout 240 --output $replay
if ($LASTEXITCODE -ne 0) { throw "Realtime replay failed" }

$r = Get-Content $replay -Raw | ConvertFrom-Json
if ($r.server_ready.independent_diarization -ne "nemotron3") { throw "Wrong backend" }
if ($r.independent_diarization.complete -ne $true) { throw "Nemotron track incomplete" }
if ($null -eq $r.final_cluster_diagnostics.anonymous_timeline) { throw "Missing caption RTTM" }

& $PY -m tools.ami_nemotron_live_score --root $root --meeting TS3003b --replay $replay --baseline $baseline --output $independent --require-under-10
$targetExit = $LASTEXITCODE
& $PY -m tools.ami_asr_stage_gap_audit --root $root --meeting TS3003b --replay $replay --output $gaps
if ($LASTEXITCODE -ne 0) { throw "ASR stage audit failed" }
Write-Host "Nemotron target exit code: $targetExit"
Write-Host "Independent report: $independent"
Write-Host "Speech-gap report: $gaps"
```

`--require-under-10` returns nonzero if the independent, full-duration,
overlap-inclusive acoustic DER is not below 0.10. The JSON is still written
for diagnosis. The old `tools.ami_live_regression_gate` checks
caption-derived DER, which is NOT the correct gate for a separate independent
acoustic timeline; keep the reports separate.

A production go/no-go also needs multi-meeting performance, calibrated
false-alarm behavior, complete model-frame coverage, no transport loss,
speaker-count error, overlap recall, ASR WER/CER against actual word
references, caption revisions and p95 latency. Never choose thresholds using
the same held-out test sample used for final acceptance.

## Tracing the missing 29.6 seconds

`MEETING_ASR_TRACE_STAGES=1` emits *timestamp-only* stage traces:

- Silero VAD intervals
- raw Whisper model timestamp intervals
- quality-verified ASR intervals
- intervals remaining after bounded recovery (if enabled)
- final confirmed caption intervals

The separate gap audit classifies each reference-speech frame with no final
caption by its earliest missing observed stage. This does not reveal what
words were omitted. That question requires source audio and a trusted word
transcript. Raw Whisper time spans can themselves be hallucinations.

## Rollback

```powershell
$env:MEETING_INDEPENDENT_DIARIZATION = "off"
$env:MEETING_ASR_TRACE_STAGES = "0"
$env:MEETING_ASR_QUALITY_CANDIDATE = "0"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "0"
```

Restart the server. The original CAM++ and Whisper path is retained.

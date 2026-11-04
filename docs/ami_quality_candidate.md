# Frozen AMI quality-candidate acceptance

This is an experimental **quality candidate**, not a validated production change.
The default behavior and original reference data are unchanged.

## What is under test

Set `MEETING_ASR_QUALITY_CANDIDATE=1` to jointly:

1. Avoid nested faster-whisper VAD inside already bounded Silero speech windows.
   The original outer Silero speech-region gate remains mandatory.
2. Keep an acoustically supported Whisper segment's original text when
   per-word alignment is incomplete or inconsistent. Do not invent words.
3. Retry up to two substantial VAD-backed untranscribed windows per inference
   call, using a real ASR decode and verifying the recovered segments.
4. Delay reclaiming VAD-backed, untranscribed buffered audio for at most two
   further recognition opportunities, then report unresolved coverage.

All changes are opt-in and use the existing speaker inference and output
contracts. Disabling the flag restores the frozen default ASR path. Candidate
retries can cost GPU time and increase latency or hallucinations. A supervised
reference text is required to assert word accuracy. The AMI RTTM is a speaker
reference, **not** a reliable transcript reference.

## Run on Windows in two PowerShell sessions

Use the project's verified environment executable, not bare `python` or
`uv` unless installed and tested. Example:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = ".\.venv\Scripts\python.exe"
& $PY -m unittest tests.quality.test_asr_quality_candidate -v
& $PY -m unittest discover -s tests -t . -v
```

First terminal (restart the server between profiles):

```powershell
$env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
$env:MEETING_ASR_QUALITY_CANDIDATE = "1"
$env:MEETING_SPEAKER_CLUSTER_STRATEGY = "bootstrap"
& $PY -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Second terminal (use a new file name for every replay):

```powershell
cd E:\Meeting-Fixed10-Test
$PY = ".\.venv\Scripts\python.exe"
$root = "benchmark/fixed195/ami_shadow10"
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$audio = "$root/audio/ami-TS3003b-1500-1740.wav"
$replay = "$root/replay_TS3003b_quality_$tag.json"
$audit = "$root/audit_TS3003b_quality_$tag.json"
& $PY -m tools.live_caption_ws_replay --audio $audio --loops 1 --speed 1.0 --output $replay
if ($LASTEXITCODE -ne 0) { throw "Replay failed" }
$r = Get-Content $replay -Raw | ConvertFrom-Json
if ($r.server_ready.asr_quality_candidate -ne $true) { throw "Wrong backend profile" }
if ($null -eq $r.final_cluster_diagnostics.anonymous_timeline) { throw "Missing timeline" }
& $PY -m tools.ami_shadow10_score --root $root --meeting TS3003b --replay $replay --output $audit
if ($LASTEXITCODE -ne 0) { throw "AMI scoring failed" }
& $PY -m tools.ami_live_regression_gate --baseline "$root/audit_TS3003b_v6_per_speaker.json" --candidate $audit --require-improvement
```

If the current clone has no `.venv`, use the already verified full
Python executable instead. Do not create or overwrite files in the AMI root;
the tools reject occupied result files.

A result may be promoted only after the explicit
`diarization_coverage_improvement_pass` is true and reference-backed text
checks, speaker count, overlap recall, hallucinations, and p95 latency are
also reviewed. `non_regression_pass` by itself is insufficient.

## Speaker clustering is a separate experiment

The independent `MEETING_SPEAKER_QUALITY_CANDIDATE=1` option:
- Lowers the minimum short-voice probe duration from 1.2 s to 0.9 s by
  default. Short probes stay untrusted unless a separate, time-aligned
  acoustic embedding corroborates them.
- Selects the existing bounded spherical clustering candidate for acoustic
  groups of 4..256 embeddings, retaining the legacy fallback.
- Performs a final speaker reassessment at session shutdown, permitting
  historical acoustic labels to be corrected, but respecting manual overrides.
  Live relabel protections remain active until shutdown.

Use the following settings in a **separate** restarted backend session:

```powershell
$env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
$env:MEETING_ASR_QUALITY_CANDIDATE = "0"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "1"
Remove-Item Env:MEETING_SPEAKER_CLUSTER_STRATEGY -ErrorAction SilentlyContinue
& $PY -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Replay and score to new result files, then confirm
`server_ready.speaker_quality_candidate=true` and
`server_ready.speaker_cluster_strategy="candidate"`. After evaluating
speaker-only and ASR-only variants independently, the combined option can be
tested, also in a new process:

```powershell
$env:MEETING_ASR_QUALITY_CANDIDATE = "1"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "1"
```

The experimental speaker model may overfragment a real speaker or still
miss heavily overlapping voices. Do not force an oracle four-person count.
The reference-backed DER, overlap recall, coverage, and latency determine
whether to retain or abandon the change.

## Known architectural limitations

- One ASR segment typically carries a single speaker label. Recovering two
  simultaneous talkers requires overlap-aware diarization and often
  source-separated ASR, not just new clustering thresholds.
- Short and heavily overlapped speech can have no trustworthy single-speaker
  CAM++ embedding. Never assign a reference speaker ID or force four clusters
  just to match the AMI evaluation case.
- This work cannot establish a lower DER from unit tests or hosted CI.
  Frozen local AMI audio and its paired ground truth must be replayed.

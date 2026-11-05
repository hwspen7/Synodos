# Pyannote Community-1 AMI 10-meeting GPU cohort

This is a **real** acoustic test, not a CPU or mock check. The ten clips
come from pinned AMI only_words references and have four minutes each.
Only one Pyannote model is loaded on the GPU; the ten separate metrics
are weighted by reference speaker-seconds.

## Run from the Windows E-drive worktree

Use a clean, current branch commit. Existing untracked `.venv-pyannote`
and `benchmark` assets need not be deleted.

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$PYP = "E:\Meeting-Fixed10-Test\.venv-pyannote\Scripts\python.exe"
$env:HF_HOME = "E:\AI-Cache"
$env:HF_HUB_CACHE = "E:\AI-Cache\hub"
$env:HF_XET_CACHE = "E:\AI-Cache\xet"
$env:TEMP = "E:\AI-Cache\tmp"
$env:TMP = "E:\AI-Cache\tmp"
New-Item -ItemType Directory -Force $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_XET_CACHE, $env:TEMP | Out-Null
& $PY -m tools.ami_shadow10_prepare --root benchmark/fixed195/ami_shadow10
if ($LASTEXITCODE -ne 0) { throw "AMI source/reference preparation failed" }

& $PYP -m tools.phase1b_pyannote_ami10 `
  --root benchmark/fixed195/ami_shadow10 `
  --reuse-report benchmark/fixed195/ami_shadow10/phase1b/pyannote_TS3003b_20261104-230436.json `
  --allow-model-download
$scoreExit = $LASTEXITCODE
if ($scoreExit -notin @(0, 2)) { throw "AMI10 GPU test did not complete" }
```

`--reuse-report` is optional. It reuses only an existing TS3003b report
whose source audio checksum, pinned reference checksum, case checksum,
model provenance, scoring protocol, CUDA device and DER components
still agree. The model runtime version is not verified for reused reports.
Omit `--reuse-report` to recompute all ten under a single model load.

Do not paste a Hugging Face access token in a console log or chat. An
already-authorized cache can be used without an environment token; a
missing or gated model may require `HF_TOKEN` and accepting the
Community-1 model conditions.

## Reading the result

The console prints ten meeting DER values and a concise
`pyannote_ami10_summary.json` in a new private folder beneath
`benchmark/fixed195/ami_shadow10/phase1b/pyannote10-*`.
Individual reports are stored in the same directory.

`0` means the entire frozen cohort's speaker-time weighted DER is below
20%. `2` means full evidence exists but the cohort gate failed;
it is **not** an inference crash. A Python exception or another nonzero
exit means incomplete evidence, and no successful full-cohort claim.

The summary separately reports whether every meeting is below 20%;
a passing weighted DER does not imply every meeting passes individually.

## Recovery from an interrupted GPU run

If interrupted, use the actual output directory shown in the Python
error or find the most recently created `pyannote10-*` directory.
The exact incomplete directory must be reused, with `--resume`.
Previously recorded reports are verified again before being trusted.

```powershell
$RUN = Get-ChildItem "benchmark\fixed195\ami_shadow10\phase1b" -Directory |
  Where-Object Name -Like "pyannote10-*" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not $RUN) { throw "No unfinished run directory found" }
& $PYP -m tools.phase1b_pyannote_ami10 `
  --root benchmark/fixed195/ami_shadow10 `
  --run-dir $RUN.FullName --resume --allow-model-download
```

If full cohort summary already exists, it is not overwritten.
A new clean run is required for new results.

## What does NOT follow from this pass

- DER is acoustic speaker diarization, not ASR WER/CER.
- Offline Pyannote latency cannot be equated with live Nemotron latency.
- The remote model weights are identified by Hugging Face repository name,
  not pinned to an immutable revision; repeatability across future downloads
  is not guaranteed without preserving that exact revision.
- A passing AMI10 Pyannote gate does not certify cross-meeting voiceprint
  FAR/FRR, correct speaker identities, model licensing for production, or
  safe automatic rewriting of human-edited transcripts.
- This runner intentionally does not change any production model defaults.

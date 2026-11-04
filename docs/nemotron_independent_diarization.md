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

## Recommended: one-command isolated Windows acceptance

The repository includes `tools/run_nemotron_ami.ps1`. It selects its own
port (8012), checks the Python interpreter, runs the targeted unit tests,
preflights cached model weights/CUDA, starts a private backend, replays the
frozen source, checks full-stream Nemotron DER and the ASR gap audit, writes
private files, and stops **only its own server process**:

```powershell
cd E:\Meeting-Fixed10-Test
& .\tools\run_nemotron_ami.ps1 -PythonExe ".\.venv\Scripts\python.exe"
```

If scripts are blocked by PowerShell execution policy, use a one-process
invocation of the reviewed repository script rather than changing global
policy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe ".\.venv\Scripts\python.exe"
```

If your verified interpreter is not under `.venv`, pass its real full path.
The script leaves V6 evidence and existing backends untouched. It returns a
nonzero exit code on any failed check or if standard Nemotron DER is >=10%.

## Alternative: two separate PowerShell terminals

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

## Investigating the 22.202% miss under the original only_words labels

NVIDIA explicitly warns that older AMI segment references sometimes count
within-segment silence as speech, inflating missed-speaker time. The model-card
AMI MHM ~9.48% DER is scored with a separate forced-aligned reference,
**not** with the project's original pinned only_words RTTM.

A standalone diagnostic compares the SAME Nemotron predictions against BOTH
reference versions. It never modifies the original audit and does NOT treat
a lower number from changed ground truth as a model improvement:

```powershell
cd E:\Meeting-Fixed10-Test
$root = "benchmark/fixed195/ami_shadow10"
$replay = "$root/replay_TS3003b_nemotron_20261104-132709.json"
if (-not (Test-Path $replay)) { throw "Original 20261104-132709 replay missing" }
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.ami_nemotron_reference_diagnostic `
  --root $root --meeting TS3003b --replay $replay `
  --output "$root/audit_TS3003b_reference_comparison_$tag.json"
```

The official-linked forced-aligned RTTM is fetched at a pinned public commit
into **E:\\Meeting-Fixed10-Test\\benchmark\\fixed195\\ami_shadow10\\forced_aligned_reference**.
No separate Python environment, NVIDIA model rerun, or OS cache path is
required. The script reports original *independent* acoustic DER and
forced-aligned *independent* acoustic DER separately, plus the caption-derived
DER as additional context. The original 23.768% result remains canonical
for the frozen current test. The new metric is a *reference-protocol
comparison*, not a deployment pass.

If both independent DERs are high, the issue is not primarily reference
boundaries; investigate the model's activity decisions, decoding model,
acoustic preprocessing, and threshold behavior on independent development
audio. Never optimize a threshold using the one held-out TS3003b sample
that will be used for final acceptance.

## Automatic paired-reference and published-caption report

The Windows one-command acceptance script now also runs the pinned
forced-alignment diagnostic on every successful replay. In addition to the
strict **original only_words** acoustic gate, it always prints and saves:

- Original-only_words independent acoustic DER.
- Forced-aligned independent acoustic DER.
- Forced-aligned acoustic DER masked to final caption clock coverage (a
  counterfactual diagnostic, not real subtitle DER).
- Forced-aligned **actual published-caption DER** and speaker count.
- The exact status of optional verified word-bound timing.

The original only_words `<10%` check remains mandatory and can still
return a **nonzero exit** even when the independent forced-aligned result is
under 10%. Both reference protocols remain separate; generated reports are
written only to private E-drive Fixed195 paths. On failure, the reports
remain available for diagnosis.

## Final speaker IDs and optional word-aligned subtitle times

The 2026-11-04 TS3003b diagnostics produced, under the **same forced
reference**, an independent acoustic DER of 7.368% but published-caption DER
of 39.591%, with 4 versus 6 published speaker IDs. The historical replay
precedes these fixes.

Two *separate* improvements now require fresh paired runs:

1. On normal meeting completion, published captions use only speaker channels
   surviving the **official** Nemotron final turn extraction. Raw 8-channel
   sigmoid activity cannot independently introduce another speaker label.
   Ambiguous official turns remain unidentified, and manual overrides win.
   This final normalization is automatic when independent diarization is
   enabled, not a ref-score trick. Intermediate live labels are provisional.
2. `-WordBounds` is an explicit ASR timing candidate. When Whisper's entire
   recognized word sequence survives verification, use the first/last valid
   word boundary rather than its broader segment boundaries. It never
   synthesizes missing words, and segments without valid words keep original
   timestamps. This may reduce no-speech false alarm but may increase gaps
   under imperfect word alignment. It is **off** by default.

The single-command test stays opt-in for each variable:

```powershell
cd E:\Meeting-Fixed10-Test
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
# A second, independent full replay with verified word-bound timing:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe" -WordBounds
```

Each run creates uniquely named reports in the E-drive Fixed195 directory.
The original only_words `--require-under-10` gate will still reject a result
at 23.768%, even if the *forced-aligned* acoustic DER is under 10%.
Do not call a changed-reference result a change in model accuracy.

For a **historical replay** produced before the fusion fix, rerun
`tools.ami_nemotron_reference_diagnostic` with a fresh output path. It now
also reports `forced_aligned_caption_clock_masked_acoustic_der`: independent
speaker activity masked to the union of existing subtitle clock intervals.
It is a controlled diagnostic to separate transcript coverage loss from
speaker-label projection, **not** a new published-caption DER, a word
accuracy claim or an end-to-end quality pass.

## Next isolated ASR-coverage experiments (2026-11-04)

The confirmed baseline is the user-run 14:02 TS3003b replay with
*forced-aligned* independent DER **7.368%**, published-caption DER **35.439%**,
and four published speakers. The separate `-WordBounds` replay at 14:08
yielded **35.731%** published DER, not an improvement. WordBounds stays OFF.

The following two candidates change different failure stages and MUST be run
one at a time, with a fresh backend and unique replay JSON for each:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"

# A. Keep outer Silero VAD; bypass only the second VAD inside each
# already-Silero-cropped auto-language Whisper window.
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe $PY -BypassWindowVad

# B. Use the prior, bounded real-ASR recovery on substantial
# VAD-positive intervals without verified transcript timestamps.
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe $PY -RecoverVadGaps
```

The runner refuses multiple ASR candidate flags in one run and keeps
`MEETING_ASR_QUALITY_CANDIDATE=0`, `MEETING_CAPTION_WORD_BOUNDS=0`,
and all other earlier regressed quality candidates OFF.

Each run reports the raw and final caption timeline, pinned only_words
DER, separately pinned forced-aligned acoustic/caption DER, and **two ASR
gap audits**. The `only_words` report is never overwritten; the forced
report is labeled `reference_protocol="forced_aligned"`, with its own
pinned reference checksum. Do NOT compare 16.4 seconds under only_words
to forced-reference missing speech without these paired audits.

The trace also counts audio inference cycles where the second VAD was
bypassed and actual gap-recovery attempts/confirmed decodes/failures.
A timestamp gap is not proof that particular human words were missing.
Compare caption text using separately verified transcripts before claiming
ASR completeness or readability.

Even if the forced-aligned acoustic DER remains 7.368%, the original
only_words strict DER <10% gate will continue to fail (23.768%). A failed
gate is not evidence that the audio replay crashed.

## Source-independent speaker support for confirmed captions

Final confirmed caption items now optionally carry
`speaker_activity_intervals_ms`, `speaker_activity_fraction` and
`speaker_activity_time_source`. These intervals come from the **official
completed Nemotron speaker turns** for the caption's selected speaker.
Unlike the original ASR sentence interval, they may contain disjoint
voiced islands, preserve quiet gaps and keep parallel speakers separate.
A caption's text, ID, revision, start/end timestamps, recorded words,
manual identity corrections and persisted ASR source remain unchanged.

The E-drive local replay JSON stores this support metadata in the final
anonymous timeline for auditing; the live UI may consume it to present
speech activity, but the original `published_caption_der` scorer still
uses the real, unchanged caption start/end timestamps. Speaker support
metadata is NOT a DER improvement or a license to delete transcript text.
Do not mark this metadata as a successful subtitle boundary fix until a
paired real-audio A/B confirms both timestamps and reading quality.

## Paired caption-coverage attribution after RecoverVadGaps

The user ran the frozen TS3003b default replay (14:02) and the candidate
RecoverVadGaps replay (14:35) against the identical forced-aligned reference.

- Default reference-active *clock* time without a caption: **12.2 seconds**.
- RecoverVadGaps reference-active *clock* time without a caption:
  **11.8 seconds** (**0.4s net gain**, not 16 individually verified captions).
- Default vs candidate forced-aligned final caption DER:
  **35.439% -> 35.029%** (0.410 percentage point improvement).
- Original-only_words final caption DER: **27.453% -> 27.775%**
  (a regression). Separate scoring protocols must never be blended.
- Across repeated inference cycles the recovery stage logged 16 candidate
  confirmations and 0 failures, not 16 independently verified transcripts.
- Nemotron's forced-aligned independent acoustic DER remained **7.368%**.

The following new report directly compares *the same reference-active
100ms clock frames* between two immutable recorded replays and discloses
gross gain **and** gross loss. Run this from E-drive:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$root = "benchmark/fixed195/ami_shadow10"
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m unittest tests.quality.test_ami_paired_caption_coverage -v
if ($LASTEXITCODE -ne 0) { throw "Paired coverage tests failed" }
& $PY -m tools.ami_paired_caption_coverage `
    --root $root --meeting TS3003b `
    --baseline "$root/replay_TS3003b_nemotron_20261104-140220.json" `
    --candidate "$root/replay_TS3003b_nemotron_20261104-143542.json" `
    --reference-protocol forced_aligned `
    --output "$root/audit_TS3003b_paired_caption_coverage_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Paired coverage comparison failed" }
```

The report also records candidate retry attempts. Recent new replays additionally
put a boolean `recovery_origin` on each entry in the transcript-free
anonymous final-caption timeline, enabling a **distinct final caption
segment count** rather than summing repeated cycle candidates. Historical
replays without this field correctly report `null` for that count. These
counts cannot verify word correctness; WER/CER still needs an independently
verified word transcript. The paired coverage comparison does not patch
captions or change the frozen original DER gate.

## Content-first speaker captions and overlapping-voice review (2026-11-04)

This is a **non-destructive, opt-in upgrade**, not a claim that every
overlapping speaker has already been independently transcribed or that
the target DER is met.

**What changed**

- `speech/content_first_projection.py` uses the final official Nemotron
  speaker timeline to attribute *individual, fully aligned Whisper words* to
  a primary speaker. If the **literal complete ASR text** cannot be
  reconstructed exactly from the word list, or its inference audio base
  timestamp is missing, the projection fails closed. Neither a word nor
  punctuation can be invented, deleted or replicated to a second speaker.
- Each canonical caption has one permanent segment ID, text, revision,
  original timestamps and translation. Extra `caption_cues` are
  **display-only**. The frontend `src/transcript/contentFirst.ts` renders
  them as split speaker lines while correction actions still target the
  original canonical segment. A manually corrected speaker invalidates any
  previous inferred speaker cue. Translations without parallel word-level
  timing are never split.
- `overlap_intervals_ms` and the final
  `content_overlap_audit` preserve independently detected simultaneous
  voice activity and highlight portions with no parallel attributed
  captions. These are **speech-activity review flags**, not fabricated text
  or proof of omitted words.
- The existing `forced_aligned_caption_der` remains the untouched,
  canonical published-caption scoring definition.
  `forced_aligned_display_projection_metrics` is reported **separately**
  only when at least one complete word-conserving split was emitted.
- `asr_coverage_trace[].cycles[].commit_receipt` accounts for candidate
  segments as newly committed, held, skipped as temporal duplicates, or
  rejected due to invalid timestamps. Counts are per-cycle, NOT WER and
  not proof of final transcript completeness. The AMI stage gap audit
  aggregates those receipts while remaining text-free.
- `MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT=1` is a **separate** existing
  quality-guard candidate exposed as `-PreserveVerifiedText`. It preserves
  the raw segment's exact text when an otherwise VAD-verified Whisper
  word alignment is inconsistent; unreliable alignment loses its word
  provenance. It is not combined with speaker projection in the single-
  variable AMI runner, pending comparison of hallucination risk and WER.
- The original audio is not automatically saved by this feature.
  Independently transcribing each simultaneous voice requires a separately
  qualified source-separation + ASR pipeline or separate microphone
  channels. Nemotron is diarization, **not** speech source separation.

**Windows isolated E-drive experiment**

```powershell
cd E:\Meeting-Fixed10-Test
$branch = (git branch --show-current).Trim()
if ($branch -ne "fix/ts3003b-integrity-forensics-20261104") {
    throw "Wrong branch: $branch"
}
git fetch origin fix/ts3003b-integrity-forensics-20261104
git merge --ff-only FETCH_HEAD
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
& $PY -m unittest tests.quality.test_content_first_projection tests.quality.test_ami_content_first_projection tests.quality.test_overlap_content_risk -v
if ($LASTEXITCODE -ne 0) { throw "Content-first contract tests failed" }
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe $PY -ContentFirst
```

The strict original `only_words` DER acceptance gate still fails at
23.768% on the historical reference even though the separate forced-
aligned acoustic DER measured 7.368%. The script prints the real, unchanged
published-caption DER and the distinct UI projection metric before
reporting the strict gate failure. No previous benchmark is replaced.

**What must be validated before production**

1. Pixel-level UI and correction/save/reload behavior with multilingual
   source text, Chinese without spaces, short turns and punctuation.
2. No confirmed word added/deleted/repeated, correct VAD-supported wording
   on held-out multilingual meetings, WER/CER and publication completeness.
3. Overlap spans that contain **two genuine simultaneous transcripts**.
   The overlap warning alone does not pass this acceptance.
4. Tail processing, streaming latency, 4070 Ti SUPER 16 GB peak VRAM,
   cross-session speaker consistency and robustness to model failure.
5. Projected speaker cue DER must remain a separately labeled metric.
   Never substitute it for canonical published-caption DER.

## TS3003b first content-first replay, actual results

The user's real Windows GPU replay `20261104-152102` completed cleanly,
and reported:

- 50 mechanically text-conserving display speaker splits;
- unchanged independent acoustic DER **7.368%** under the separately
  pinned forced-aligned RTTM;
- canonical final caption DER **35.497%** under that same forced reference;
- distinct *UI-projected* speaker-cue DER **32.515%**, with missed speaker
  time increasing from **17.544% to 25.146%** and false alarms dropping
  from **16.023% to 6.725%**. This is not a WER improvement or evidence
  that no speaker words were lost. The best earlier canonical DER remains
  **35.029%** from a different RecoverVadGaps replay; do not treat it as
  a direct display-projection baseline.
- official acoustic overlap full-speaker recall **64.286%** while
  canonical final caption overlap full-speaker recall remained **0%**;
- acoustic review clock **10,270ms**, of which **7,666ms** was marked
  underrepresented by *canonical* parallel caption labels. This is
  under-attributed speaker activity, not confirmed missing words.
- commit receipts across 299 inference cycles: 514 verified candidates,
  438 held, 76 first-time committed, zero duplicate skips and zero
  invalid timestamp rejections. Held events recur across cycles and
  cannot be interpreted as 438 lost utterances. No AMI WER/CER exists
  in this replay.

A subsequent diagnostic correction now exports both
`content_overlap_audit.overlap_clock_underrepresented_ms` (canonical
caption activity) and
`content_overlap_audit.actual_ui_display_projection.overlap_clock_underrepresented_ms`
(actual speaker display cues). The projection **replaces** its canonical
parent in this review-only calculation to prevent double-counting
voices. Neither definition counts missing *words*.

To compare canonical versus actual UI-displayed overlapping-speaker
activity on this **already completed** replay, without invoking CUDA or
re-running the audio, use:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
& $PY -m unittest tests.quality.test_content_first_projection tests.quality.test_ami_offline_display_overlap -v
if ($LASTEXITCODE -ne 0) { throw "Content-first audit tests failed" }
$root = "benchmark/fixed195/ami_shadow10"
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.ami_offline_display_overlap `
    --root $root --meeting TS3003b `
    --replay "$root/replay_TS3003b_nemotron_20261104-152102.json" `
    --output "$root/audit_TS3003b_display_overlap_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Offline overlap audit failed" }
```

This new offline audit reads and validates the immutable original AMI source
with `score_case`, computes canonical and actual-display acoustic overlap
under identical model output, and refuses to silently overwrite a report.
Its measurements are **not** double-stream transcribed word coverage.

The valid next acceptance requires held-out multilingual transcription
references to evaluate WER/CER and missed words, plus source-separated
overlap speech samples with independent word annotations. Do not
enable `-ContentFirst` by default until these pass.

## Rollback

```powershell
$env:MEETING_INDEPENDENT_DIARIZATION = "off"
$env:MEETING_ASR_TRACE_STAGES = "0"
$env:MEETING_ASR_QUALITY_CANDIDATE = "0"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "0"
```

Restart the server. The original CAM++ and Whisper path is retained.

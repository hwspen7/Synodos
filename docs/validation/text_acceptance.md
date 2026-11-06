# Phase 1 transcript integrity and release acceptance

This document separates independent reference-backed text accuracy from
diarization DER and from engineering correctness. An absent ground-truth
transcript is **not** a passing CER/WER result.

## Locked Phase 1 product scope: ten languages

Official Phase 1 supported recognition languages:
`en, zh, fr, es, ar, id, ja, ru, de, hi`.
Portuguese (`pt`) and Bengali (`bn`) have been removed from the
product-supported selection, not erased from the existing recordings
or frozen evaluation archive. The supported language menu and both
the meeting settings API and WebSocket enforce the ten-language list.
`auto` is permitted for source-language detection, but detection is
not a guarantee that Whisper will never recognize an out-of-scope
language. Regardless of the detected language, the recognized text
is not discarded. Translation destinations are limited to ten
languages or disabled; the actual offline language pair must still
be installed.

The original Fixed195 archive remains **130 single-speaker samples
in 12 languages** and its existing historical acceptance report remains
available. The new *product* gate reads the complete pinned original
130-case ASR result, verifies original provenance and scores **110
selected samples in ten approved languages**. This is an explicitly
narrowed product scope, NOT measured model improvement. The two
excluded sets remain archived and must never be deleted to produce
a passing score.

On the same full, code-fingerprint-matching `content_review_singles_*.json`
that was already evaluated, run the new selection gate (no GPU required):

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$root = "benchmark/fixed195"
# The scorer creates both a raw report and a newer *_summary.json.
# Select the complete case-level report, not its aggregate-only summary.
$inputReport = Get-ChildItem "$root/content_review_singles_*.json" -File |
    Where-Object { $_.Name -notmatch '(_summary|_triage)\.json
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.validation.language_gate `
    --root $root `
    --results $inputReport.FullName `
    --output "$root/phase1_10_languages_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Ten-language accuracy acceptance failed" }
```

If you previously selected a generated `*_summary.json` by mistake,
the ten-language scorer resolves it to the exact paired raw JSON when
present, but still requires the raw report to pass schema, audio,
reference, model and code-fingerprint checks. It never scores a summary
directly or reconstructs missing case-level evidence.

The general engineering script now runs this selected ten-language gate
when `-Fixed195Results` is supplied. The legacy
`tools.validation.fixed195_text_gate` intentionally still evaluates
all twelve historical languages and may report a failure for Bengali;
that does **not** mean the new ten-language product selection failed.

## What the new release checks actually prove

- `tools/maintenance/replay_transcript_score.py` measures human-reference CER and
  space-delimited WER, now including separately attributed deletion,
  insertion and substitution counts under a deterministic minimum-edit
  alignment. Deletion means reference text absent from the final hypothesis
  under that alignment; it is not proof about why the recognizer omitted it.
  Character error remains supported for CJK without fabricated word breaks.
- `tools/validation/text_acceptance.py` requires a complete WebSocket replay,
  exact pinned original-audio and reference-text hashes, manual confirmation
  that the reference was independently checked, no disappearing previously
  confirmed captions, no silent confirmed-text revision, and bounded CER,
  WER and deletion rates. Defaults are illustrative *candidate engineering*
  thresholds, not established product performance: CER <= 10%, space-based
  WER <= 20% where meaningful, and reference word/character deletion <= 3%.
  If no transcript has been manually verified, this gate cannot pass.
- `tools/validation/fixed195_text_gate.py` re-scores 130 existing Fixed195
  single-speaker cases with their frozen human references. It checks each
  audio SHA256, all case IDs, reference identity and ASR model/code
  fingerprint before measuring total/individual-language CER and relative
  deletion. Default gates: weighted CER <= 15%, weighted character
  deletions <= 5%, each source language CER <= 25%, all cases completed,
  no empty predictions. These are candidate thresholds only.
  Multi-speaker samples without human text references are **not scored**.
- A source transcript remains the only canonical saved segment, even when
  the UI shows multiple word-attributed speaker cues. Regression tests now
  cover real `MeetingStore` save/reload/JSON export of the same source
  text, without multiplying source records.
- `tools/run_engineering_acceptance.ps1` runs all model-free/backend contract
  tests, the entire Python test suite, Vue type-check, production frontend
  build, Node transcript/WebSocket tests, and Windows PowerShell syntax
  parsing, with package/cache directories on the E: drive. It does not
  run live GPU inference or itself establish speech correctness.

## 1. Local engineering acceptance (no GPU needed)

From PowerShell 5.1, on the project worktree:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin fix/ts3003b-integrity-forensics-20261104
git merge --ff-only FETCH_HEAD
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_phase1_engineering.ps1 -PythonExe $PY
```

If `node_modules` is already complete, `-NoFrontendInstall` skips
package installation. Never point any output into C:. If compilation,
a test, type-check or build fails, the script stops without declaring
engineering success.

## 2. Reuse all 130 original references and score the 110 selected Phase 1 clips

If the prior pinned run is already present, reuse it. Otherwise run
`tools.evaluation.corpora.fixed195_content_review` against the frozen single manifest:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$root = "benchmark/fixed195"
& $PY -m tools.evaluation.corpora.fixed195_content_review --root $root --mode singles --model large-v3 --device cuda --compute-type float16
if ($LASTEXITCODE -ne 0) { throw "Fixed195 inference failed" }
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.validation.language_gate --root $root --results "$root/content_review_singles.json" --output "$root/phase1_10_languages_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Ten-language, 110-clip text quality not accepted" }
```

When the result file exists, you can run only the second tool. If the
existing ASR report is an older fingerprint, use a newly named output
with `tools.evaluation.corpora.fixed195_content_review`, preserving the previous report.
It checks
the model/source/code fingerprint, so older or modified-code measurements
may be rejected rather than silently reused. This test can run with
`run_phase1_engineering.ps1 -Fixed195Results "$root/content_review_singles.json"`
to combine code checks and the reference-backed text gate.

## 3. Check final live-caption fidelity on the same referenced audio

The existing TS3003b diarization RTTM has **speaker-time labels**, not a
complete verified word transcript. Do not use the RTTM as an ASR text
reference. Obtain and independently check a true full transcript of the
same `ami-TS3003b-1500-1740.wav` excerpt first. Save that private UTF-8
text inside `benchmark/fixed195/ami_shadow10/`. Then use:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$reference = "benchmark\fixed195\ami_shadow10\TS3003b_human_verified.txt"
$sha = (Get-FileHash $reference -Algorithm SHA256).Hash.ToLowerInvariant()
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe $PY -ContentFirst -HumanReferenceText $reference -ExpectedReferenceSha256 $sha -HumanReferenceVerified
```

The runner now passes `--reference-text` to the real one-pass WebSocket
replay, computes exact private CER/WER/deletion metrics, emits the pinned
reference and hypothesis digests without plaintext, then runs the
independent Phase 1 human text gate. It also prints the original frozen
only_words DER separately and will still return a failure when the
original strict <10% DER acceptance gate is not met. **A passing text
gate must never be confused with passing that separate diarization gate.**
If the verified transcript is absent, run without the new reference
flags and treat text accuracy as *not measured*.

## 4. GitHub CI and release rule

Three GitHub workflows exist: backend/frontend/PostgreSQL, operations,
and Windows launcher/parser. Added explicit Phase 1 content contracts to
the main backend CI and PowerShell parsing of both acceptance scripts
to the Windows workflow. The job scheduler for the most recent observed
GitHub Actions attempts returned failures with zero executed steps and
no usable test logs. A retry of all three also failed before job
execution. That cannot be repaired by relaxing assertions or modifying
the source tree; the repository owner must resolve GitHub Actions
runner availability/account policy/billing if applicable.

Do **not** tag or merge Phase 1 as passing until:
1. The local engineering script passes, plus hosted CI executes and passes.
2. At least the 130 complete human-referenced single recordings pass the
   pinned text gate, and the end-to-end live meeting text gate passes
   with a genuinely verified transcript of the corresponding audio.
3. Manual listening finds no obvious missed *intelligible* complete
   utterances, especially in short interjections, despite aggregate CER/WER.
4. No source text disappears, is fabricated or duplicated during edit,
   reconnect, save/reload or export.

Neither a low speaker DER nor overlap activity recall is a substitute
for human-reference text fidelity.
 } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $inputReport) { throw "Full 130-clip ASR source report is required" }
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.validation.language_gate `
    --root $root `
    --results $inputReport.FullName `
    --output "$root/phase1_10_languages_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Ten-language accuracy acceptance failed" }
```

The general engineering script now runs this selected ten-language gate
when `-Fixed195Results` is supplied. The legacy
`tools.validation.fixed195_text_gate` intentionally still evaluates
all twelve historical languages and may report a failure for Bengali;
that does **not** mean the new ten-language product selection failed.

## What the new release checks actually prove

- `tools/maintenance/replay_transcript_score.py` measures human-reference CER and
  space-delimited WER, now including separately attributed deletion,
  insertion and substitution counts under a deterministic minimum-edit
  alignment. Deletion means reference text absent from the final hypothesis
  under that alignment; it is not proof about why the recognizer omitted it.
  Character error remains supported for CJK without fabricated word breaks.
- `tools/validation/text_acceptance.py` requires a complete WebSocket replay,
  exact pinned original-audio and reference-text hashes, manual confirmation
  that the reference was independently checked, no disappearing previously
  confirmed captions, no silent confirmed-text revision, and bounded CER,
  WER and deletion rates. Defaults are illustrative *candidate engineering*
  thresholds, not established product performance: CER <= 10%, space-based
  WER <= 20% where meaningful, and reference word/character deletion <= 3%.
  If no transcript has been manually verified, this gate cannot pass.
- `tools/validation/fixed195_text_gate.py` re-scores 130 existing Fixed195
  single-speaker cases with their frozen human references. It checks each
  audio SHA256, all case IDs, reference identity and ASR model/code
  fingerprint before measuring total/individual-language CER and relative
  deletion. Default gates: weighted CER <= 15%, weighted character
  deletions <= 5%, each source language CER <= 25%, all cases completed,
  no empty predictions. These are candidate thresholds only.
  Multi-speaker samples without human text references are **not scored**.
- A source transcript remains the only canonical saved segment, even when
  the UI shows multiple word-attributed speaker cues. Regression tests now
  cover real `MeetingStore` save/reload/JSON export of the same source
  text, without multiplying source records.
- `tools/run_engineering_acceptance.ps1` runs all model-free/backend contract
  tests, the entire Python test suite, Vue type-check, production frontend
  build, Node transcript/WebSocket tests, and Windows PowerShell syntax
  parsing, with package/cache directories on the E: drive. It does not
  run live GPU inference or itself establish speech correctness.

## 1. Local engineering acceptance (no GPU needed)

From PowerShell 5.1, on the project worktree:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin fix/ts3003b-integrity-forensics-20261104
git merge --ff-only FETCH_HEAD
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_phase1_engineering.ps1 -PythonExe $PY
```

If `node_modules` is already complete, `-NoFrontendInstall` skips
package installation. Never point any output into C:. If compilation,
a test, type-check or build fails, the script stops without declaring
engineering success.

## 2. Reuse all 130 original references and score the 110 selected Phase 1 clips

If the prior pinned run is already present, reuse it. Otherwise run
`tools.evaluation.corpora.fixed195_content_review` against the frozen single manifest:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$root = "benchmark/fixed195"
& $PY -m tools.evaluation.corpora.fixed195_content_review --root $root --mode singles --model large-v3 --device cuda --compute-type float16
if ($LASTEXITCODE -ne 0) { throw "Fixed195 inference failed" }
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
& $PY -m tools.validation.language_gate --root $root --results "$root/content_review_singles.json" --output "$root/phase1_10_languages_$tag.json"
if ($LASTEXITCODE -ne 0) { throw "Ten-language, 110-clip text quality not accepted" }
```

When the result file exists, you can run only the second tool. If the
existing ASR report is an older fingerprint, use a newly named output
with `tools.evaluation.corpora.fixed195_content_review`, preserving the previous report.
It checks
the model/source/code fingerprint, so older or modified-code measurements
may be rejected rather than silently reused. This test can run with
`run_phase1_engineering.ps1 -Fixed195Results "$root/content_review_singles.json"`
to combine code checks and the reference-backed text gate.

## 3. Check final live-caption fidelity on the same referenced audio

The existing TS3003b diarization RTTM has **speaker-time labels**, not a
complete verified word transcript. Do not use the RTTM as an ASR text
reference. Obtain and independently check a true full transcript of the
same `ami-TS3003b-1500-1740.wav` excerpt first. Save that private UTF-8
text inside `benchmark/fixed195/ami_shadow10/`. Then use:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
$reference = "benchmark\fixed195\ami_shadow10\TS3003b_human_verified.txt"
$sha = (Get-FileHash $reference -Algorithm SHA256).Hash.ToLowerInvariant()
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_nemotron_ami.ps1 -PythonExe $PY -ContentFirst -HumanReferenceText $reference -ExpectedReferenceSha256 $sha -HumanReferenceVerified
```

The runner now passes `--reference-text` to the real one-pass WebSocket
replay, computes exact private CER/WER/deletion metrics, emits the pinned
reference and hypothesis digests without plaintext, then runs the
independent Phase 1 human text gate. It also prints the original frozen
only_words DER separately and will still return a failure when the
original strict <10% DER acceptance gate is not met. **A passing text
gate must never be confused with passing that separate diarization gate.**
If the verified transcript is absent, run without the new reference
flags and treat text accuracy as *not measured*.

## 4. GitHub CI and release rule

Three GitHub workflows exist: backend/frontend/PostgreSQL, operations,
and Windows launcher/parser. Added explicit Phase 1 content contracts to
the main backend CI and PowerShell parsing of both acceptance scripts
to the Windows workflow. The job scheduler for the most recent observed
GitHub Actions attempts returned failures with zero executed steps and
no usable test logs. A retry of all three also failed before job
execution. That cannot be repaired by relaxing assertions or modifying
the source tree; the repository owner must resolve GitHub Actions
runner availability/account policy/billing if applicable.

Do **not** tag or merge Phase 1 as passing until:
1. The local engineering script passes, plus hosted CI executes and passes.
2. At least the 130 complete human-referenced single recordings pass the
   pinned text gate, and the end-to-end live meeting text gate passes
   with a genuinely verified transcript of the corresponding audio.
3. Manual listening finds no obvious missed *intelligible* complete
   utterances, especially in short interjections, despite aggregate CER/WER.
4. No source text disappears, is fabricated or duplicated during edit,
   reconnect, save/reload or export.

Neither a low speaker DER nor overlap activity recall is a substitute
for human-reference text fidelity.

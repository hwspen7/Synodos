# System-wide Whisper pipeline diagnosis

Fifteen priority languages yield 68 original single-speaker recordings.
All 100 original clips cover 23 language codes. These small samples are not
a statistically representative survey of meetings, speakers or microphones.

## Shared architecture

- WhisperEngine calls Silero VAD prior to decoding.
- Default window-LID splits long continuous speech into nonoverlapping windows
  capped at 18 seconds. It may bisect words in uninterrupted speech.
- A second Whisper VAD runs on each audio window.
- verify_utterances filters low-VAD-support segments and words.
- The sentence splitter reconstructs recognized word text.
- Explicit Korean word-separator loss has a conservative independent fix.

## Commands

Run from the repository root with your existing CUDA environment.

Pilot the 15-language group (partial results exit with code 2 by design):

    .\.venv\Scripts\python.exe tools\fixed150_system_audit.py --scope focus --limit 3

Complete all 68 priority-language recordings:

    .\.venv\Scripts\python.exe tools\fixed150_system_audit.py --scope focus

Audit all 100 original recordings, including the non-priority languages:

    .\.venv\Scripts\python.exe tools\fixed150_system_audit.py --scope all

Three controlled paths use the same Whisper large-v3 model and frozen audio:

1. Full production windowed automatic language detection
2. Full WhisperEngine with whole-clip automatic language detection
3. Direct faster-whisper auto-decoding, without app-specific text rejection

Ground-truth forcing is not used in any of these paths. The results test
system-level regressions but cannot independently prove the double-VAD stage
is causal. Source-only reference-review flags remain in all raw totals.

Outputs are local: benchmark/fixed150/system_ablation_focus.json and
system_ablation_focus_summary.json, or corresponding _all files.
Progress is checkpointed case by case. A fingerprint checks model, source
audio SHA, and related pipeline code before reusing existing measurements.

Each row includes weighted CER ingredients, paired differences, language
and audio duration, detected-language labels, latency, number of segments
and potential 18-second seams within VAD-detected speech. No raw model
transcript or source reference is stored; the originals are never modified.

## Interpreting the data

- If windowed CER exceeds whole-clip CER, suspect external segmentation.
- If whole-clip CER also exceeds raw CER, inspect text verification and VAD.
- If all three fail, audit reference fidelity and test another model.
- Positive and negative differences both matter. Never tune only to one clip.
- 5 percentage-point paired deltas flag candidates, not conclusive causality.
- 50 multispeaker clips need an independent streaming and overlap evaluation.

Do not globally disable per-window language detection until code-switched
and overlapping speech are validated with strict timestamp de-duplication.

## Stage-level ablation result: 8 impacted originals

The 2026-11-03 user-run stage audit showed that switching internal Whisper
VAD off is **not** uniformly better. In Thai case 071, raw CER improved
from 26.45% (internal VAD on) to 8.26% (off), but Thai case 069 worsened
from 25.27% (on) to 56.99% (off). Polish 043 and Portuguese 047 improved
when turning it off. Stage-level word verification worsened a previously
perfect Thai case 072 from 0% raw CER to 11.69%, regardless of the
internal-VAD setting, and Portuguese 045 worsened from 3.28% to 9.02%.

The code therefore **does not** turn internal VAD off globally.

The shared text guard has a narrow optional experiment. If the full
transcription segment passes acoustic and decoding checks, but its word
timestamps do not reliably align with speech intervals, this mode retains
the whole segment text with segment-level timestamp quality, rather than
deleting some words. It does not fabricate word timestamps, ignores no
diagnostic checks, and leaves the production default strict:

    MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT=0

A reproducible regression test validates that unsupported speech and
high no-speech-probability segments remain rejected.

To test the policy without changing the deployed default, run the updated
stage audit on the *same* eight impacted originals:

    .\.venv\Scripts\python.exe tools\fixed150_stage_audit.py --scope all

This now writes to a **new** checkpoint:
benchmark/fixed150/stage_guard_ablation.json.

Per VAD setting, the report compares three results for each recording:
raw Whisper output, original strict text verification, and opt-in
evidence-preserving verification. Only the original 8 affected clips are
tested, so these metrics cannot prove an overall improvement. They are
used to decide whether to run a full 100-original and mixed-language
regression, which must precede enabling the opt-in production flag.

Do not overwrite the original stage_audit.json or frozen source manifest.

## Full 100-case matched-output guard audit

The eight-case alignment-preservation trial had four improvements, two
regressions, and two unchanged clips with internal Whisper VAD enabled.
The eight cases had already been selected for poor original performance.
These figures cannot estimate the benefit across the full dataset.

Evaluate all 100 original single-speaker clips, including healthy samples:

    .\.venv\Scripts\python.exe tools\fixed150_guard_audit.py --scope all

Only one Whisper inference is run per acoustic window. Identical model
segments are scored using raw text, the strict guard, and the experimental
segment-preserving guard. This isolates policy effects from re-decoding.

The numeric checkpoint is benchmark/fixed150/guard_audit_all.json.
The separate summary is benchmark/fixed150/guard_audit_all_summary.json.
Both are local. Raw transcript strings are not saved. Each case has
reference-review flags and is preserved in the original CER denominator.
Fifteen priority languages are reported separately. An interrupted run
can resume if the source hashes, model name and relevant code match.

Do not enable the experimental production mode based on preselected bad
examples or a favorable aggregate alone. Verify held-out normal speech,
silence/noise, bilingual code switching, speaker overlap and live captions.
Do not globally disable secondary VAD: Thai 069 clearly regressed.

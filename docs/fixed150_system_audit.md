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

# Fixed10 multilingual benchmark and minimal meeting workspace

## Scope and evidence boundary

This is an initial, **ten-clip immutable regression set**, not a proof of
support for every mainstream language. It deliberately avoids selecting only
English and Mandarin. A measured 85/100 release requires a much larger,
consented, annotated set, GPU inference, stability testing and verified speaker
attribution. Ten clips are necessary for fast iteration, not sufficient for
certifying the six Phase 1 defects.

The current sample allocation:

| Case | Locale or condition | Source | What it tests |
| --- | --- | --- | --- |
| 01 | English (United Kingdom) | PolyAI MInDS-14 | Latin token spacing, recognition |
| 02 | Mandarin Chinese | PolyAI MInDS-14 | Hanzi transcription, auto LID |
| 03 | Spanish | PolyAI MInDS-14 | Orthography and accents |
| 04 | French | PolyAI MInDS-14 | Apostrophes, elision |
| 05 | German | PolyAI MInDS-14 | Compounds and segmentation |
| 06 | Korean | PolyAI MInDS-14 | Hangul and Unicode |
| 07 | Portuguese | PolyAI MInDS-14 | Language identification |
| 08 | Russian | PolyAI MInDS-14 | Cyrillic and Unicode |
| 09 | 3 simultaneous speakers, different European languages | LAION | Speaker-overlap stress |
| 10 | 3 simultaneous speakers, different European languages | LAION | Independent overlap replication |

The mix languages in 09/10 are taken from the dataset ground truth instead of
being invented. Actual source files, source revisions, corpus license, SHA-256
WAV checksums, source rows, references and sample durations are frozen in
`lock.json` **after** download and conversion, then verified at every run.

Source repositories:

- https://huggingface.co/datasets/PolyAI/minds14
- https://huggingface.co/datasets/laion/small-overlapping-speech-bench

Both source datasets identify CC-BY-4.0 licensing; retain upstream attribution
in research reports and redistributions. LAION's test data is based on
Multilingual LibriSpeech (Pratap et al., Interspeech 2020).

## Windows: install dependencies and freeze WAVs

Use the **flattened project root** from the feature branch, **not**
`backend/backend` or `frontend/frontend` from the old `main` layout.
Never apply these files on top of a dirty local working tree.

```powershell
# Run from the flattened meeting repository root.
& .\.venv\Scripts\python.exe -m pip install huggingface_hub pyarrow soundfile scipy numpy plotly
& .\.venv\Scripts\python.exe tools\fixed10_prepare.py --root benchmark\fixed10
& .\.venv\Scripts\python.exe tools\fixed10_prepare.py --root benchmark\fixed10 --verify-only
```

The first command requires network access and can download hundreds of
megabytes because MInDS parquet shards are much larger than the selected
single audio. For decoder wheels that do not support MP3 on Windows, install a
compatible libsndfile build or adapt the decoder to verified FFmpeg.
Do not change or overwrite a locked WAV in place: create a new benchmark
version when the dataset must change.

## Run actual GPU-based ASR inference

First verify the model and dependency setup in the current Python environment
(see `tools/doctor_06c.py`). Run only one primary ASR at a time unless
measuring deliberate concurrency. `large-v3` is the initial baseline, not a
certified winner.

```powershell
& .\.venv\Scripts\python.exe tools\phase07_benchmark.py --manifest benchmark\fixed10\manifest_single.jsonl --output benchmark\fixed10\single_asr.json --infer --engines whisper --whisper-model large-v3
& .\.venv\Scripts\python.exe tools\fixed10_dashboard.py --root benchmark\fixed10
Start-Process .\benchmark\fixed10\dashboard.html
```

The benchmark can use `--engines whisper qwen3` only after installing and
validating the optional Qwen3-ASR dependency. It writes actual CER,
language-aware token error and latency. `phase07_benchmark` measures full
audio-call inference time, not live end-to-end microphone latency.
**No baseline or comparison scores are included in this branch.** The
dashboard will not show an invented metric when results are unavailable.
A failed sample is a blocker, not a zero-error sample.

To measure **all ten clips** rather than only the eight unambiguous
single-speaker ASR samples, replay all ten WAVs through a running meeting
backend. The tool verifies that the server acknowledged the exact PCM sample
count and returned a completed transcript snapshot. Audio is sent at its
actual wall-clock rate by default (1.0×).

```powershell
# In one terminal, start the validated backend
& .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# In another terminal, run from the project root
& .\.venv\Scripts\python.exe -m pip install websockets
& .\.venv\Scripts\python.exe tools\fixed10_replay.py --root benchmark\fixed10 --url ws://127.0.0.1:8000/ws/audio
& .\.venv\Scripts\python.exe tools\fixed10_dashboard.py --root benchmark\fixed10
```

The replay saves `replay_results.json` for all ten cases and
`overlap_predictions.json` for the complete pipeline's two overlapping
clips. Transport errors, incomplete acknowledgements and missing finalization
are flagged as failures. WebSocket replay does not substitute for testing
the physical microphone, browser audio journal or multi-hour reliability.

For external models you may alternatively supply timestamped predictions:

```json
{
  "fixed10-09": {
    "segments": [
      {"start_ms": 0, "end_ms": 3250, "speaker_id": "speaker_1", "text": "verbatim model output"}
    ]
  },
  "fixed10-10": {
    "segments": []
  }
}
```

These are **schema examples**, not test results. Replace them entirely with
actual predictions. File path:
`benchmark/fixed10/overlap_predictions.json`. The dashboard computes
speaker-label permutation-invariant DER and per-person content-word recall.
Missing predictions appear as unmeasured and block overlap acceptance.
Use one globally stable speaker ID per independent recording. If the audio is
unidentifiable, retain the unknown speaker rather than inventing an identity.

## Charts and decision gates

Seven visual panels: (1) CER/model/language heatmap, (2) CER vs mixed-token
error, (3) inference latency, (4) language ID accuracy, (5) overlapping
speaker content-word recovery, (6) source-verified acoustic speaker occupancy,
(7) review/blocked/pass evidence table. The generated HTML bundles Plotly JS
for offline reading and uses the existing orange/teal product palette.

Exploratory smoke thresholds **set before measurements**: CER <= 22%,
mixed-token error <= 32%, automatic language identification >= 75%, and
inference p95 <= 3500 ms. These are not universal multilingual thresholds.
In particular, an eight-clip aggregate cannot establish statistical
confidence. A smoke pass is NEVER a release pass, nor a claimed Zoom score.
Live delay, tail-word recall, audio loss, burst overload, diarization,
resumable recording and crash integrity must be scored separately.

If results fail, apply the failure-specific changes:

1. Language-identification errors: evaluate per-utterance LID confidence,
   detect language shifts, use multilingual language routing. Do **not**
   force all recordings to English, Mandarin or a per-meeting constant.
2. High CER/token error: compare Whisper large-v3, Qwen3-ASR and applicable
   specialist engines **on identical WAVs**, and inspect deletions, noise,
   accents, script-aware normalization and cold-start latency.
3. Repeated/missing words: track immutable audio sample intervals, acoustic
   evidence and word-alignment provenance; never fix by blind text deletion.
4. High diarization DER: independently benchmark continuous acoustic turns,
   pyannote Community-1 for offline refinement and Sortformer for streaming
   up to its supported speakers. Preserve unknown identities.
5. High p95 or queue starvation: isolate speech, background diarization,
   translation and Saluki workloads with bounded priority queues, GPU budget
   and backpressure observability.
6. Dropped audio or missed ending: implement durable browser-side PCM
   journal + resumable ACK/replay, stronger crash recovery and sample
   conservation tests. The current branch does **not** claim those repairs.

Model changes require paired before/after reports. Do not promote a candidate
because of its publication date or benchmark claims from a different system.

## Production validation beyond Fixed10

Extend to FLEURS (102 languages), recent Common Voice and a consented
30+ clip true meeting corpus. Include Arabic, Hindi, Japanese, Turkish,
Vietnamese and additional languages missing from this fast set. Cover
code-switching, noisy/quiet microphones, short acknowledgements,
A-B-A diarization, overlap, long silence and short final words. Hold out a
never-used evaluation split to avoid tuning on the ten regression clips.
Run two 60-minute recordings and a 2-hour soak with active Saluki on Windows
and record source SHA, Git commit, VRAM, p50/p95 response delays, reconnect
outcomes and audio sample equality.

## UI architecture

The meeting page separates foreground actions from expert configuration.

- Main area: meeting title, recording state, stop/start, save/export,
  brief source-language and translation state, timestamped transcript.
- Secondary settings: participant enrollment, hotword libraries, language,
  ASR window and diagnostics. It does not claim unsupported Argos language
  pairs are installed.
- Optional review side pane: summary, contextual corrections and offline
  speaker refinement.

The main palette stays orange/teal from `src/assets/main.css`. The layout
borrows dense but focused controls from mainstream meeting applications
without copying proprietary screens. Unknown speaker and missing translation
states remain explicit. Automatic language detection is the default only for
new meetings or meetings without saved language preferences. Translation is
off by default so original transcript is the source of truth.

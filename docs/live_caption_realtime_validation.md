# Live subtitle reliability: first production-safe improvement

The Fixed195 single-speaker reference-backed baseline (130 completed cases,
8.409% pooled CER; 8.1388% language-macro CER) is intentionally frozen.
This change does not modify Whisper's VAD thresholds, beam search, word
alignment verification, or the seven-second minimum commit context.

## Changes

- Provisional Whisper caption inference now triggers at the requested
  preview interval (default 2.5 seconds of buffered audio) instead of
  silently being clamped to 4 seconds. The complete time to first visible
  caption also includes inference time, thread scheduling and WebSocket/UI.
- Provisional caption revisions are monotonic. An explicit empty partial
  clears a draft only once after it becomes confirmed or after final ASR.
- Speaker reclustering, translation updates and new confirmed snapshots
  no longer silently erase still-unconfirmed text. Revisions from an old
  or duplicate partial event are ignored in the browser.
- Live provisional captions are rendered **inside** the transcript panel
  as separate display-only sentences, visually labeled as not confirmed.
  Only confirmed segments enter persisted meeting transcripts.
- Speaker labels remain unidentified for provisional captions. The
  system does not fabricate a speaker attribution before acoustic
  diarization completes.

## Local Windows validation

Update the local checkout:

    cd E:\Meeting-Fixed10-Test
    git fetch origin feature/multilingual-fixed10-ui-refresh
    git switch --detach origin/feature/multilingual-fixed10-ui-refresh

**First test manually through the existing UI.** Start the backend and
frontend as usual, permit microphone access, and speak two short sentences
with a natural pause. Confirm that the draft appears in the transcript
panel before the seven-second commitment window and is replaced by durable
segments after confirmation. Check that changing speaker labels never
causes remaining draft text to disappear. Stop the recording and confirm
the final utterance is present in the saved transcript.

For an instrumented replay of authentic two-person audio, start the
backend in one PowerShell window (unless it is already running):

    cd E:\Meeting-Fixed10-Test
    .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

In a second PowerShell window, replay an existing real DataHive sample at
**1.0x real-time** with the same versioned PCM transport as the browser:

    cd E:\Meeting-Fixed10-Test
    .\.venv\Scripts\python.exe tools\live_caption_ws_replay.py --audio benchmark\fixed195\audio\fixed195-en-m01.wav --loops 1 --speed 1

Review the private local report:

    benchmark\fixed195\live_caption_replay.json

It reports first provisional/confirmed caption wall times, transcript
versions, speaker reassignment count, final confirmed segment count,
the server's audio sample acknowledgement, and whether stop/finalization
completed. The first caption may appear later than the configured
2.5-second buffer interval due to inference and transport overhead.
The report deliberately contains no raw transcripts.

For an **endurance-only** run, repeat the same authentic 75-second
conversation 25 times in a single WebSocket session:

    .\.venv\Scripts\python.exe tools\live_caption_ws_replay.py --audio benchmark\fixed195\audio\fixed195-en-m01.wav --loops 25 --speed 1 --output benchmark\fixed195\live_caption_25loops.json

Repeated audio is NOT 25 independent conversations and cannot be used to
claim speech accuracy. It tests long-session buffer turnover, WebSocket
delivery, monotonically ordered snapshots and stop/finalization. For a
short smoke test first use `--loops 1`.

## Interpretation

For user-visible provisional captions, the measurement target is low
time-to-first-draft and no disappearance between updates. **No target
latency is asserted as measured until a local 1.0x replay is completed.**

The readiness checks require:
- `delivery_matched=true` and server-received samples equal samples sent.
- `server_closed=true` and no error events.
- Zero nonmonotonic partial or transcript revision flags.
- First draft, confirmed captions and final saved text examined in the UI.
- No duplicate confirmed turns from a partial-to-final handoff.

Only checks of transport and chronology are automatic in the replay.
Transcript completeness, intelligibility and correct diarization during
speaker overlap still require listening with independently annotated
ground truth. Multi-language and overlapping speech quality are not
certified by these changes.

### Optional experimental latency configuration

The backend supports `MEETING_PREVIEW_INTERVAL_S=1.5` for an isolated
local comparison. This increases the frequency of speculative Whisper
calls and could change draft stability. Do not treat a faster preview as
a certified improvement until measured on an independent recording.
Production default remains 2.5 seconds.

## First local 75-second A/B outcome (November 3, 2026)

A real-time local DataHive English 75-second dialogue was replayed twice
using the identical audio transport. After switching to the current
backend and verifying the 2.5s processing window, observed metrics were:

| Metric | Before (4.0s window) | After (2.5s window) |
|---|---:|---:|
| First provisional subtitle wall latency | 5277.1ms | 3364.7ms |
| First confirmed subtitle wall latency | 8645.2ms | 7626.5ms |
| Non-monotonic subtitle revision flags | 6 | 0 |
| Final confirmed segments | 25 | 24 |
| Cumulative speaker label reassignments | 54 | 48 |
| Received PCM samples | 1,200,000 | 1,200,000 |

Both runs received all the sent PCM and completed the WebSocket session.
The follow-up emitted 8 `feature_warning` events. Prior reporter versions
count warnings but did not show their feature or message. This is a
sample-level observation, not a statistical latency guarantee; 48 speaker
ID changes are **snapshot transition counts**, not 48 proven incorrect
speakers, and 24 vs 25 text segments is not a proven accuracy change.

## Run the enhanced speaker/warning triage

After updating this branch in your existing test checkout, rerun the same
75-second real-time audio command. The report now additionally contains:

- `feature_warning_counts` by feature plus up to 12 short distinct
  `feature_warning_samples`. These remain in the local Git-ignored report.
- `speaker_reassignment_types` categorizing transitions between currently
  unidentified, assigned cluster/label, or a different assigned label.
  **Labeled does not necessarily mean a known person's identity.**
- `segments_with_speaker_changes` and
  `max_speaker_changes_one_segment` to distinguish routine one-time
  initial assignment from repeated back-and-forth relabels.
- `confirmed_text_revisions` and
  `confirmed_segment_disappearances` to flag unexpected instability
  within complete transcript snapshots, without treating all legitimate
  post-processor splits as failures.

The diagnostics do not change the model, speaker cluster assignments,
decoding parameters, or backend cadence. These outputs supplement, not
replace, visual/listening checks of the full transcript. Repeated playback
of one clip is insufficient to certify diarization quality.


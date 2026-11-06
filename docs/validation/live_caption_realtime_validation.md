# Current diarization policy — restored stable baseline (November 2026)

The current recommended backend setting is `MEETING_DIARIZATION_MODE=cluster`.
The default audio-to-text path is unchanged. Voice identification is an
optional, best-effort annotation on confirmed text, **not** a prerequisite
for publishing subtitles.

- Keep the existing global acoustic cluster backend. Never specify an
  oracle speaker count just to make a two-person test pass.
- Preserve the existing 120-second horizon for embedding-backed, non-uncertain
  historical speaker labels through
  `MEETING_SPEAKER_LABEL_REVISABLE_S=120`. Recent labels may still be
  corrected by new evidence; manual overrides remain authoritative.
- When speaker embedding is absent or unusable, publish the recognized
  text with `Unidentified Speaker`; do not discard a sentence or
  manufacture an identity based on neighboring labels.
- Retire the `cluster_anchors` and `cluster_sticky` experiments. An
  obsolete Windows environment value now resolves safely to `cluster`,
  and WebSocket `ready.diarization_mode` reports the effective mode.
  The explicit `centroid` and `single` options are retained only for
  compatibility / controlled tests; neither is recommended as default.
- Keep `CLUSTER_MERGE_THR=0.78`, `MEETING_PREVIEW_INTERVAL_S=2.5`,
  and `MEETING_MIN_COMMIT_CONTEXT_S=7.0` unless independently evaluated.
  This restoration does **not** claim diarization DER or Zoom-equivalent
  recognition quality.

On a Windows backend shell, stop the previous server and explicitly set:

```powershell
$env:MEETING_DIARIZATION_MODE = "cluster"
$env:MEETING_SPEAKER_LABEL_REVISABLE_S = "120"
$env:CLUSTER_MERGE_THR = "0.78"
$env:MEETING_PREVIEW_INTERVAL_S = "2.5"
$env:MEETING_MIN_COMMIT_CONTEXT_S = "7.0"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Run a one-loop, 75-second replay before treating the rollback as verified
locally. Check `server_ready.diarization_mode` equals `cluster`, all
PCM samples arrive, `flags`/`errors` are empty, and captions remain
complete. Distinct labels and relabel churn remain diagnostic, not
ground-truth speaker accuracy. Don't run additional speaker-architecture
experiments without labeled turns and a core transcription impact.

**Historical notes below are archived diagnostics, not active run
instructions.** Their use of `cluster_sticky` or `cluster_anchors`
describes now-retired experiments.

---

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



## Long-session Windows playback pacing and speaker stabilization

A user-ran 25-loop 1.0x 31.45-minute replay delivered all 30,192,000
PCM samples, ended without WebSocket errors, and produced 625 confirmed
segments. It also exposed two separate issues:

1. The first replay implementation slept for 0.2 seconds **after each**
   transmitted frame. The Windows event loop added roughly 4.6ms/frame
   of overhead. With 9,447 frames, actual wall time exceeded 1,887 seconds
   of recorded audio by about 43.6 seconds. The reported apparent subtitle
   lag increased at almost the same rate. This was predominantly a **replay
   clock artifact**, not evidence by itself of backend queue growth.
2. The all-history speaker reclustering pipeline changed existing positive
   speaker IDs 5,991 times over the run, with up to 53 changed assignments
   for a single confirmed segment. These were snapshot identity churn
   measurements, NOT 5,991 ground-truth diarization mistakes.

The replay script now paces each frame relative to the absolute
\`first_send + total_samples / sample_rate\` monotonic clock deadline.
Sleeping to absolute deadlines avoids cumulative scheduling oversleep.
The report adds \`sender_pacing_drift_ms\`, a separate measurement of
playback lag **before** final ASR and WebSocket shutdown, plus
\`max_late_send_ms\` and \`expected_replay_wall_ms\`. If the sender itself
lags by more than two seconds, the report flags the measurement rather
than attributing its lag trend to the backend.

A conservative streaming speaker-label stability policy is now enabled in
cluster mode. Raw cluster labels are aligned globally using overlap with
previously displayed speaker IDs. Historical, voiceprint-backed,
non-uncertain speaker assignments older than 120 seconds are preserved
against subsequent relabels. Newly confirmed and uncertain segments remain
mutable. Manual speaker corrections remain authoritative. This policy
reduces visible historical churn at a potential cost to retrospective
diarization corrections; it does NOT certify correct speaker identities.

For direct comparison, set \`MEETING_SPEAKER_LABEL_REVISABLE_S=0\` before
starting the backend to disable history freezing. The new overlap matching
still operates in both configurations, so this is not an exact binary
reproduction of the old code. The default is 120 seconds; values 0..3600
are valid. Centroid and forced-single-speaker modes are unchanged.

### Short acceptance replay before another 31-minute run

From \`E:\Meeting-Fixed10-Test\`, restart the backend so its new speaker
label policy loads. Then use the latest CLI to run:

    .\.venv\Scripts\python.exe tools\live_caption_ws_replay.py --audio benchmark\fixed195\audio\fixed195-en-m01.wav --loops 4 --speed 1 --output benchmark\fixed195\live_caption_4loops.json

Review \`sender_pacing_drift_ms\`, \`timing_by_elapsed_window\`,
\`speaker_reassignment_types\`, \`max_speaker_changes_one_segment\`,
\`delivery_matched\`, \`flags\` and \`errors\`. Only advance to 25 loops
after the sender clock is close to actual audio time and the backend
completes its final transcript without protocol errors. Compare audio,
model, language, segment counts and source conditions across runs.

Long-run user-facing diarization quality, sentence completeness and
overlapped speech still require independent annotations and listening.


## Experimental speaker-fragmentation root-cause diagnosis

The 301.5-second four-loop A/B on the same two-person conversation
found that `cluster_sticky` produced 10 distinct published speaker labels,
down from 20 for the seeded `centroid` option, but still more than the
two actual source speakers. There were zero positive-label-to-positive-label
reassignments and no transport errors. These results do not establish
diarization accuracy.

Do not promote the freeze-on-first-label behavior or blindly reduce
the centroid similarity threshold. Both can hide a mistaken identity.

The final raw-cluster count was previously absent from the replay report.
That prevented distinguishing a genuine acoustic-clustering split from
historical ID aliases created by label freezes. Run a diagnostic replay
with the existing opt-in `cluster_sticky` mode and:

    $env:MEETING_DIARIZATION_DIAGNOSTICS = "1"

Start the backend with this environment variable set. On the final
WebSocket `closed` event, the server includes a bounded, anonymous
`diarization_diagnostics` object. The replay tool records it under
`final_cluster_diagnostics`.

Interpretation:

- `raw_cluster_count` close to the known number of voices but
  `published_positive_speaker_count` substantially higher means
  identity mapping or historical label freezing has fragmented the UI.
- Both counts substantially higher than the source voice count means
  the acoustic clustering itself is likely over-splitting.
- `raw_clusters_with_multiple_published_labels` and
  `max_published_labels_per_raw_cluster` quantify how many stored
  labels correspond to the same raw cluster, without automatically
  conflating those labels or making an accuracy claim.

The diagnostic contains only aggregate counts and raw ephemeral cluster
indexes, no recognized text, audio, speaker embeddings or participant
names. It is intentionally opt-in, and changes neither diarization mode
nor the spoken-content processing pipeline. A future merging algorithm
must still be evaluated with known speaker turns and overlap cases.


## November 2026 A/B root cause: raw acoustic over-splitting AND ID aliases

On the fourth 301.5-second repeated-dialogue replay with
\`cluster_sticky\` and diagnostics enabled, the source contained two real
speakers but the end-of-session report showed:

| Diagnostic | Result |
|---|---:|
| FunASR raw acoustic clusters | 10 |
| Distinct published positive speaker labels | 8 |
| Raw clusters with multiple published labels | 8 |
| Published labeled segments within conflicted raw clusters | 85 |
| Segments without voice embeddings | 5 |
| Positive-labeled ID-to-ID changes | 0 |
| PCM delivery and final WebSocket close | Complete |

Two independent problems remain: the acoustic clustering generated more
groups than the true number of voices, and the sticky alias preservation
retained inconsistent IDs across most groups. Freezing earlier is not an
identity-accuracy fix. Do not claim a diarization error rate from these
aggregate counts.

### One-run, read-only merge sensitivity

The opt-in \`MEETING_DIARIZATION_DIAGNOSTICS=1\` final report now also
contains \`configured_merge_threshold\` and \`posthoc_merge_sweep\`.
The sweep re-merges the *existing* final acoustic cluster centroids
at several lower cosine thresholds without changing the transcript,
speaker assignments, or model configuration.

To obtain an exact audio-matched comparison, rerun the same 4-loop replay
after updating this branch and restarting the backend with diagnostics
enabled. Then view:

    $r = Get-Content benchmark\fixed195\live_caption_sticky_diagnostic.json -Raw | ConvertFrom-Json
    $r.final_cluster_diagnostics.posthoc_merge_sweep | ConvertTo-Json -Depth 6

This sweep estimates which thresholds would consolidate obvious acoustically
similar subclusters. It is NOT a new FunASR spectral clustering run and
cannot prove whether the merged clusters correspond to real people. A
configuration change is not justified until the speaker-turn assignment is
checked against human-labeled reference intervals, including overlaps.


## Short reference-aligned speaker turn inspection (75-second clip)

The current posthoc merge sweep still produces four groups at a cosine
threshold of 0.55 from the ten final raw groups. A smaller group count
alone cannot establish correct assignment to the two real speakers.

Use **one loop** of the original 75-second audio for speaker-turn
verification, not four concatenated repeats. With the backend's
\`MEETING_DIARIZATION_DIAGNOSTICS=1\` option set, the final
\`diarization_diagnostics.anonymous_timeline\` contains chronological
\`start_ms\`, \`end_ms\`, \`raw_cluster_id\`,
\`published_speaker_id\`, \`embedding_available\` and
\`speaker_uncertain\`. It exports at most 256 segments and reports
truncation explicitly. No transcript content, names or embeddings are
included in the report.

    .\.venv\Scripts\python.exe tools\live_caption_ws_replay.py --audio benchmark\fixed195\audio\fixed195-en-m01.wav --loops 1 --speed 1 --output benchmark\fixed195\live_caption_turns_1loop.json

    $r = Get-Content benchmark\fixed195\live_caption_turns_1loop.json -Raw | ConvertFrom-Json
    $r.final_cluster_diagnostics.anonymous_timeline.segments |
        Format-Table start_ms, end_ms, raw_cluster_id, published_speaker_id, embedding_available, speaker_uncertain -AutoSize

Align those intervals against the pinned local speaker reference
annotations, if available. The existing \`tools/evaluation/corpora/fixed150_prepare.py\`
stores time-coded two-speaker AMI \`reference_segments\` in the local
\`benchmark/fixed150/lock.json\`; \`fixed195\` is not tracked in git,
so DO NOT assume a particular reference layout exists for its WAV.
Without reference turns, this output is a diagnostic timeline, not
speaker identification accuracy or DER. The final talker-identity test
must include overlaps and mixed-speaker sentences.

This anonymous timeline and the merge sweep are opt-in diagnostics.
The default production clustering, speech recognition and provisional
subtitle cadence remain unchanged.


## November 2026 short-recording A/B and anchor-only cluster experiment

The user ran the original English two-person 75-second WAV **once** with
the default \`cluster\` mode. At finalization all 27 confirmed segments
had a usable speaker embedding, FunASR produced exactly two raw
clusters, the UI showed two positive IDs, and there were no final
cluster-to-ID conflicts. However, *during* the short meeting there
were 35 positive-ID-to-positive-ID changes (57 speaker updates total).
Correct final speaker *count* is not proof of correct speaker-turn
identity. Ground-truth reference alignment remains necessary.

In contrast, four repeats of the same 75-second audio had 10 final
raw clusters and eight published IDs under \`cluster_sticky\`, suggesting
that growth of acoustic evidence changes FunASR's cluster structure.
The code path for the acoustic clustering is identical between
\`cluster\` and \`cluster_sticky\`, but this is a separate run with a
different experimental presentation policy. The difference cannot be
called a causal proof or a speaker diarization quality score.

To test the hypothesis that very short/mixed voice windows seed
spurious acoustic clusters, a new **opt-in** backend mode
\`MEETING_DIARIZATION_MODE=cluster_anchors\` now:

1. Groups only embedding-backed, non-uncertain speech segments of at
   least \`MEETING_ANCHOR_MIN_DURATION_S\` (default 1.2 seconds) with
   the existing FunASR clustering backend. It handles a single
   trusted long segment without invoking the spectral eigensolver.
2. Assigns a shorter or uncertain segment to an existing raw cluster
   only if its embedding has at least cosine similarity
   \`MEETING_ANCHOR_SHORT_SIMILARITY\` (default 0.72) and has an
   unambiguous best-match margin
   \`MEETING_ANCHOR_SHORT_MARGIN\` (default 0.08).
3. Keeps the segment's transcript but marks the speaker unidentified
   when acoustic evidence is missing or ambiguous. Short audio
   **cannot create an additional speaker ID** by itself.
4. Uses the same existing historical ID mapping. Existing
   \`cluster\`, \`centroid\`, and \`cluster_sticky\` modes are unchanged.

This is an experimental acoustic identity-creation policy, not a
validated solution; its primary quality trade-off is a potential
increase in unidentified short speech. In synthetic tests two anchor
speakers stay separated, tiny outliers cannot seed new speaker
clusters, and uncertain short speech is left unidentified. These
tests do not estimate live DER or actual speaker correctness.

Run an A/B with **one** 75-second loop first, then four loops only if
the single-loop outcome does not break identity count or coverage:

    $env:MEETING_DIARIZATION_MODE = "cluster_anchors"
    $env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
    $env:MEETING_PREVIEW_INTERVAL_S = "2.5"
    .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

    .\.venv\Scripts\python.exe tools\live_caption_ws_replay.py --audio benchmark\fixed195\audio\fixed195-en-m01.wav --loops 1 --speed 1 --output benchmark\fixed195\live_caption_anchors_1loop.json

Compare \`final_assigned_speaker_count\`,
\`final_unidentified_segment_count\`,
\`final_cluster_diagnostics.raw_cluster_count\`,
\`labeled_to_different_labeled\` transitions, \`flags\`,
\`errors\` and the anonymous speaker timeline against a human speaker
reference where possible. **Do not simply select the mode with the
smallest number of speaker IDs.** Leave default production mode at
\`cluster\` until validated.


## Core backend turn boundary improvement — November 2026

The main Whisper speech recognition settings, default \`cluster\` speaker
mode, existing 120-second historical speaker label policy and caption
confirmation window have not changed.

The existing \`split_by_acoustic_turns\` helper now inspects roughly
1.0-second acoustic windows for an intra-utterance speaker handoff.
Earlier 1.4-second windows could not examine two distinct voices within
a roughly 2.0–2.7 second ASR span. A stricter 0.45 similarity gate
applies to the shorter windows to reduce false changes from normal
variation in one voice. At most eight embeddings are calculated for
one confirmed phrase, as before.

- Turn splits require authentic, ordered word timestamps and a complete
  source-text match, including every word and punctuation mark. If word
  alignment is missing or even slightly inconsistent, the full original
  confirmed phrase is retained. Never guess a speaker change timestamp.
- When an acoustic change is found, the confirmed phrase may become
  multiple accurately timed subtitle records. Embeddings are calculated
  conservatively from voice-consistent window evidence. Very short or
  ambiguous speech retains the unidentified speaker label rather than
  guessing another person's identity.
- The original source text remains the authority: no punctuation, words
  or speech content may be invented or omitted during acoustic splitting.
- Experimental \`centroid\` and rejected \`cluster_sticky\` /
  \`cluster_anchors\` diarization paths are **not** re-enabled.
- With opt-in \`MEETING_DIARIZATION_DIAGNOSTICS=1\`, final WebSocket
  diagnostics and the replay JSON include
  \`final_cluster_diagnostics.acoustic_turn_splits\`, the number of extra
  transcript records created by confirmed acoustic turn splits. This is
  a diagnostic count, **not** speaker identification accuracy or DER.

Test one 75-second replay, after updating and restarting the backend:

\`\`\`powershell
$env:MEETING_DIARIZATION_MODE = "cluster"
$env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
$env:MEETING_PREVIEW_INTERVAL_S = "2.5"
.\\.venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
\`\`\`

\`\`\`powershell
.\\.venv\\Scripts\\python.exe tools\\diagnostics\\live_caption_ws_replay.py --audio benchmark\\fixed195\\audio\\fixed195-en-m01.wav --loops 1 --speed 1 --output benchmark\\fixed195\\live_caption_short_turns.json
$r = Get-Content benchmark\\fixed195\\live_caption_short_turns.json -Raw | ConvertFrom-Json
$r.final_cluster_diagnostics.acoustic_turn_splits
$r | Select-Object final_confirmed_segments, final_unidentified_segment_count, speaker_reassignment_types, delivery_matched, flags, errors
\`\`\`

The expected improvement is recognition of actual quick speaker changes
*when reliable timed words exist*, with no lost source text or transport
regression. Synthetic regression tests validate a 2.4-second two-person
utterance, absent/inconsistent word alignment, acoustic continuity,
mixed voice evidence and snapshot preservation. Real speaker-turn
accuracy and ASR completeness still require reference-labeled audio.


## Opt-in local ASR accuracy against a verified human transcript

A 75-second real two-person sample can transfer every PCM sample and
still omit or hallucinate words. The standard
\`tools/diagnostics/live_caption_ws_replay.py\` JSON deliberately does not contain
recognition strings; \`asr_cer: null\` means **not reference-scored**.
The ungated English \`fixed195-en-m01.wav\` from the public dialogue
collection has **no independently verified transcript** in that
audio-only collection. Never manufacture a reference from the ASR
hypothesis or interpret a low segment count as accuracy.

To export a **local-only** text hypothesis once, run a single loop
after updating the checkout and restarting the backend:

\`\`\`powershell
.\\.venv\\Scripts\\python.exe tools\\diagnostics\\live_caption_ws_replay.py --audio benchmark\\fixed195\\audio\\fixed195-en-m01.wav --loops 1 --speed 1 --output benchmark\\fixed195\\live_caption_text_audit.json --transcript-output benchmark\\fixed195\\local_hypothesis_en_m01.txt
\`\`\`

The transcript export is **explicit opt-in**, limited to new \`.txt\`
files inside the gitignored \`benchmark/fixed195/\` directory. The
normal JSON report, console and CI never reveal recognized words.
It refuses to overwrite an existing private export.

A human must listen to the same WAV independently, correct the entire
spoken content including interjections and other speakers, and save
\`benchmark/fixed195/human_reference_en_m01.txt\` as a UTF-8 file.
Copying the hypothesis to the reference without listening yields a
meaningless 0% edit rate. Overlapping speech may not have a unique
linear order; if so, assess overlap manually as well.

Score the already-exported ASR hypothesis *offline* with no additional
audio replay:

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m tools.maintenance.replay_transcript_score --hypothesis-text benchmark\\fixed195\\local_hypothesis_en_m01.txt --reference-text benchmark\\fixed195\\human_reference_en_m01.txt --output benchmark\\fixed195\\local_text_score_en_m01.json
\`\`\`

Alternatively, specify \`--reference-text\` directly on a new
\`tools/diagnostics/live_caption_ws_replay.py --loops 1\` run, which populates
\`asr_cer\`, \`asr_wer\` and \`reference_scoring\` in the aggregate
JSON. Reference scoring with repeated loops is deliberately rejected.

Metrics are transparent and aggregate-only: NFKC/casefold CER excludes
punctuation/whitespace; punctuation-sensitive CER preserves punctuation
but removes whitespace; English whitespace-token WER excludes
punctuation. Unsegmented CJK WER is reported as \`null\` rather than an
invented tokenization score. CER and WER are edit counts divided by
reference length and can exceed 100%. These metrics alone do not
measure speaker assignment, overlap attribution, word timestamps or
a reference's independent validity.

This local validation is the next core acceptance gate. Compare
missing words/repeated words and false turn boundaries against
actual audio before retuning acoustic thresholds. If final text
completeness is below target, fix the confirmed-text source first;
do not trade content for prettier or smaller subtitle segments.

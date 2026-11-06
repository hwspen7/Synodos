# Phase 08 Repair Status

This register tracks the 36 issues found during the November 2026 code review.
A green unit-test run is not a real-world accuracy certification. Do not merge
or deploy as an authenticated multiuser service until the remaining P0 risks are addressed.

## Status definitions

- **Mitigated** means a targeted implementation change and regression coverage exist, but the real-world outcome still needs validation.
- **Partial** means only one component or failure mode was improved.
- **Open** means no adequate end-to-end solution is in place.

| ID | Audit issue | Status | Notes |
| --- | --- | --- | --- |
| 01 | Automatic language detection replaced by English | Mitigated | Explicit auto passes through WebSocket unchanged |
| 02 | Duplicate committed transcript windows | Partial | Conservative time-and-text overlap guard; intra-window ASR hallucinations remain |
| 03 | Whole-utterance speaker embeddings | Open | Speaker-turn-aligned diarization needs dedicated audio model |
| 04 | Browser WebSocket audio congestion | Partial | Bounded outstanding send bytes with explicit stop/error; no resume protocol |
| 05 | English preference versus Chinese legacy backend | Partial | Whisper is now default; optional runtime dependency must be installed |
| 06 | Recording transport and reconnection recovery | Open | Confirmed transcript resumes; live unconfirmed audio does not |
| 07 | Overlapping speakers | Open | Existing offline analysis does not feed authoritative speaker transcripts |
| 08 | API user authentication and authorization | Open | Local host/origin limits remain; no production identity layer |
| 09 | Latin-script paragraph spacing | Mitigated | Frontend and export share equivalent script-aware formatting rules |
| 10 | Short-segment speaker identity | Open | Embeddings below 1.2 seconds are not extracted |
| 11 | Frequent speaker reclustering | Open | Stable speaker IDs can still be reassigned after new evidence |
| 12 | Uncalibrated speaker similarity | Open | No controlled DET/EER or diarization benchmarks |
| 13 | Whisper short-speech suppression | Open | VAD thresholds need empirical false-rejection study |
| 14 | Qwen output validation parity | Open | Model adapters do not share all confidence checks |
| 15 | Fixed maximum ASR window boundaries | Open | No cross-window word reconciliation |
| 16 | Translation pair coverage | Open | Installed Argos language-pair models determine availability |
| 17 | Empty no-translation request overridden | Mitigated | Explicit empty target no longer falls back to English |
| 18 | Translation runs in ASR processor | Open | Needs separate bounded translation worker |
| 19 | Only confirmed text periodically persisted | Partial | Opt-in WAV has periodic readable header/disk checkpoints |
| 20 | Background worker stale-lease recovery | Partial | Periodic stale-job scan and live-job heartbeat; distributed failure drills outstanding |
| 21 | Entire PostgreSQL meeting rewritten on save | Open | Incremental persistence schema/workflow needed |
| 22 | Legacy pickle deserialization | Open | Need versioned structured format and safe migration |
| 23 | Incomplete meeting deletion | Partial | Meeting-owned WAV and PostgreSQL AI jobs/artifacts/translations/knowledge are deleted; shared media excluded |
| 24 | Claim citation validity versus source support | Open | IDs are validated, semantic entailment is not |
| 25 | Recent-only retrieval candidates | Open | Need full-corpus hybrid index and tests |
| 26 | Browser resampling quality | Partial | Cross-block fractional phase corrected and sample-count regression added; linear resampling still lacks anti-alias filter |
| 27 | Very small audio frames | Open | Worklet packet batching and scheduling needed |
| 28 | UI Clear does not erase persisted transcript | Open | Need explicit clear-versus-delete controls |
| 29 | Word-aligned audio playback corrections | Open | Edit by segment ID exists but no synced review timeline |
| 30 | Two translation state pipelines | Open | Need canonical translation store and display contract |
| 31 | Diagnostics do not measure correctness | Partial | Model/diarization mode visible; quality depends on external labels |
| 32 | Summary sentence-level citations | Open | Summary claim evidence graph needed |
| 33 | Limited translation target options | Open | User interface does not discover installed language models |
| 34 | Outdated README | Mitigated | Documentation updated for optional AI, audio and model dependencies |
| 35 | Incomplete model/audio provenance | Partial | Selected ASR model and diarization mode shown; no recording-bound model manifest |
| 36 | End-to-end multilingual benchmark | Open | Existing unit tests and offline benchmark cannot replace annotated GPU recordings |

## High-priority acceptance gates

1. **Accuracy:** Collect consented recordings with separate ground-truth transcripts and speaker turns. Include several supported languages, code-switching, dialects, background noise and cross-talk. Report per-language character/word error rate and diarization error rate. No unverified vendor accuracy claims.
2. **Temporal correctness:** Confirm all speech time ranges are monotonic and that genuinely repeated spoken phrases are preserved. Test repeated model output and out-of-order confirmations separately.
3. **Audio durability:** Inject process crashes at random points during 60-minute recordings. Verify readable committed WAV data, detected recovery gaps and complete confirmed-text snapshots.
4. **Security:** Keep deployment limited to localhost until user authentication, authorization, CSRF policy, secret handling and safe legacy data migration have been validated.
5. **Real-time metrics:** Measure end-to-end subtitle latency, P50/P95 GPU inference time, queue starvation, sustained memory usage and network backlog with real models.
6. **Speaker validation:** Compare clustering versus an independent diarization pipeline using at least two speakers and overlapping speech. Audio mixed into one mono channel is inherently harder than remote participant-isolated tracks.

## Privacy and backward compatibility

- Do not silently convert or erase existing recordings and transcripts.
- No bundled model downloads or forced migration are included in these commits.
- Preserve original recognized text when translating to another display language.
- Any automatic output suppression should require acoustic and temporal evidence, not just unusual wording.
- Optional WAV recording is disabled unless the user enables it. The browser's audio transport can still lose its unacknowledged tail during an unexpected shutdown.

## Local validation

Run:

```powershell
cd E:\Personal_Project\meeting
git status -sb
git pull --ff-only origin refactor/flatten-project-root
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
pnpm type-check
pnpm build
```

Install the optional Whisper runtime into the active virtual environment before
running the new default recognizer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-speech-whisper.txt
```

If this installation uses a Python environment without pip, use
`uv pip install --python .\.venv\Scripts\python.exe -r requirements-speech-whisper.txt`.

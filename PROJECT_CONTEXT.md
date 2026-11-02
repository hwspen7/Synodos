# Meeting of Pioneers — Technical Project Context

**Updated:** November 2, 2026

This document is intended for developers and coding assistants working on Meeting of Pioneers. It describes behavior observable from the tracked source code, not a guarantee of fully tested production behavior. The source code and developer-facing text are in English. Chinese speech recognition remains supported.

## 1. Product overview

Meeting of Pioneers is a local-first web application for recording meetings through a browser microphone, transcribing speech with punctuation and timestamps, assigning speaker identities, and saving or exporting transcripts.

Three kinds of user data are stored locally:

1. **Meetings:** metadata, confirmed transcript segments, timestamps, speaker labels, embeddings, and recording configuration.
2. **Speakers:** enrolled identities with multiple voiceprint embeddings per person.
3. **Hotwords:** named vocabulary lists selected for specific meetings.

Implemented functionality includes meeting creation and management, live transcription, speaker diarization, voiceprint enrollment from live recording or existing meeting segments, participant matching, hotword libraries, resumable sessions, and Markdown/JSON export.

Not currently implemented: automatic summaries, action-item extraction, full-text search, audio recording playback, manual transcript editing, system-audio capture, user authentication, database storage, multi-tenant isolation, or cloud synchronization.

## 2. Technology and source layout

**Frontend:** Vue 3, TypeScript, Vue Router, Vite, pnpm, browser AudioWorklet, WebSocket.

**Backend:** Python 3.12+, FastAPI, uvicorn, FunASR, PyTorch, NumPy, file-based pickle persistence, uv.

```text
meeting/
├── main.py                     FastAPI setup, routers, startup preload
├── api/
│   ├── health.py               Health and diagnostics
│   ├── meetings.py             Meeting CRUD, save, and export
│   ├── speakers.py             Speaker and voiceprint operations
│   ├── hotwords.py             Hotword library operations
│   └── websocket.py            Live audio protocol and connection lifecycle
├── core/
│   ├── config.py               Settings, thresholds, and model paths
│   └── runtime.py              Shared model/store objects and active sessions
├── audio/buffer.py             Thread-safe PCM buffer
├── inference/models.py         ASR, CAM++ embeddings, speaker clustering
├── sessions/
│   ├── session.py              Per-connection Session
│   └── processor.py            Per-session transcription thread
├── storage/
│   ├── meetings.py             Meeting pickle snapshots and transcript export
│   ├── speakers.py             Speaker identity and embedding storage
│   └── hotwords.py             Hotword library storage
├── example/                    Manual testing scripts
├── tests/                      Basic route and persistence regression tests
├── src/
│   ├── App.vue, main.ts        Frontend application root
│   ├── pages/                  Meeting, speaker, and hotword screens
│   ├── components/             Shared visual components
│   ├── api/client.ts           HTTP API client
│   ├── api/ws.ts               Realtime WebSocket client
│   ├── audio/                  Microphone capture and PCM conversion
│   ├── speaker/                Speaker display helpers
│   ├── router/                 Vue Router configuration
│   └── assets/main.css         Shared styles
├── public/
├── index.html, vite.config.ts
├── package.json, pnpm-lock.yaml
├── pyproject.toml, uv.lock
└── README.md
```

Backend code is organized by responsibility rather than placed in a monolithic `app/` package. New HTTP endpoints should be implemented in `api/`; model-specific logic belongs in `inference/`; persistence in `storage/`; connection or transcription lifecycle in `sessions/`.

## 3. User workflows

### Meetings

The home screen lists stored meetings and displays aggregate meeting, duration, segment, and speaker counts. The aggregate speaker number is a sum across meetings, not a deduplicated global identity count.

Creating a meeting persists initial metadata and opens its meeting view. Existing meetings can be viewed, resumed, deleted, or exported. Opening a meeting may establish a WebSocket connection to restore saved segments even before a new microphone recording begins.

The meeting view allows switching between segment and paragraph display, selecting known participants and hotword libraries, adjusting matching and processing thresholds, saving, exporting, and copying transcript text.

### Speaker enrollment

A speaker can hold multiple voiceprint vectors. A vector may originate from a microphone recording or a confirmed segment of an existing meeting. Each vector retains duration, source, and optional meeting and segment references. Samples shorter than the configured minimum are flagged as potentially less reliable.

### Hotwords

Hotword libraries store names and lists of terms. The selected libraries are combined and deduplicated for a meeting, with a limit of 200 terms supplied to ASR. Hotword matching is a recognition hint, not a guarantee that specific terms will always be produced.

## 4. Audio and transcription pipeline

```text
Browser microphone
  -> AudioWorklet downmix, resampling, Int16 PCM
  -> WebSocket /ws/audio
  -> Session -> AudioBuffer
  -> SessionProcessor background thread
  -> FunASR transcription, VAD, punctuation, sentence timestamps
  -> Sentence confirmation and CAM++ voiceprint extraction
  -> Selected participant cosine matching
  -> Clustering of unmatched segments
  -> Full transcript snapshot to browser
  -> Segment coloring and paragraph grouping in Vue

Save / stop -> MeetingStore -> meeting.pkl
Export -> Saved transcript -> Markdown or JSON
```

The browser requests microphone capture with echo cancellation, noise suppression, and automatic gain control. Audio is downmixed/resampled to **16,000 Hz mono signed Int16 little-endian PCM without a WAV header**. The backend normalizes PCM into floating-point samples by dividing by 32768. The WebSocket does not decode compressed audio formats.

Recognition is **windowed near-real-time ASR**, not character-by-character streaming:

1. By default, after 10 seconds of buffered audio, the processor runs ASR on the audio window.
2. The result contains sentences, text, punctuation, and millisecond timestamps.
3. A sentence is confirmed only if its ending precedes the buffered window end by at least 2 seconds, except during finalization.
4. A CAM++ embedding is extracted once for each confirmed sentence.
5. Confirmed audio is trimmed from the rolling window; unconfirmed audio remains for subsequent processing.
6. Once confirmed, sentence text and embeddings are not recomputed just because speaker clustering changes.

`VAD_THRESHOLD_S` is the processing-buffer duration, not a voice probability. `REC_THRESHOLD_S` is a trailing-audio confirmation delay, not an ASR confidence level. Under the default 10-second buffer trigger, pausing for 2 seconds does not guarantee immediate transcript display.

When stopping, the client sends the JSON control `{"type":"stop"}` after microphone capture stops. The backend finalizes remaining buffered audio without requiring the usual confirmation delay, but discards very short final buffers below the configured threshold (default 1 second). It then saves the session and sends the final transcript and close notification.

Resuming restores confirmed segments, embeddings, and participant/hotword settings. The next recording starts on the internal sample timeline after the last confirmed segment; elapsed real-world time between recording sessions is not added to transcript timestamps.

## 5. Model management and speaker identity

| Model alias | Role |
| --- | --- |
| `paraformer-zh` | Chinese-capable ASR |
| `fsmn-vad` | Voice activity detection and segmentation |
| `ct-punc-c` | Punctuation restoration |
| `cam++` | Speaker embedding extraction |

`inference/models.py` owns lazy model initialization and transcription/embedding/clustering calls. Startup also eagerly preloads the models through a FastAPI startup hook. Embeddings are flattened to float32 and L2-normalized. The code does not explicitly select a compute device; actual CPU/GPU execution depends on the runtime environment.

**Known participants:** For each confirmed sentence, the embedding is compared to all enrolled vectors belonging to participants explicitly selected for this meeting. The best cosine similarity determines the match when it reaches the threshold. The backend does not implicitly search every enrolled speaker globally.

**Unmatched speakers:** Remaining segments are passed to FunASR ClusterBackend for re-clustering. Stable speaker IDs are approximated by mapping old and new clusters using shared segment membership. Speaker labels can still change after clustering updates and do not establish cross-meeting identity.

The default participant matching threshold is **0.3**. This is separate from the default clustering merge threshold **0.78**. The UI can override the participant threshold.

When participants are selected, unmatched segments use labels such as `Unknown Speaker 1`. Otherwise they use labels such as `Speaker 1`. Source-code defaults have been translated to English; previously stored labels and user-supplied speaker names are not retroactively rewritten.

**Paragraphs:** Adjacent segments are grouped when the `speaker_id` matches and the gap is strictly less than `merge_gap_s` (2 seconds by default). The frontend groups segments for display; the backend performs equivalent grouping during export. This does not cause additional ASR.

## 6. Concurrency and state

The FastAPI event loop handles HTTP and WebSocket traffic. Every WebSocket connection has one `Session` and one `SessionProcessor` worker thread. `AudioBuffer` uses synchronization primitives for concurrent append/snapshot/trim operations.

Sessions share a single `ModelManager` and a global inference lock. Consequently, ASR, embedding extraction, and clustering inference are serialized across concurrent sessions. Each session forwards worker messages through an asyncio queue with capacity 500.

`core/runtime.py` holds the `sessions` registry keyed by session ID and the `meeting_sessions` registry keyed by meeting ID. A new connection for an active meeting signals the old processor to stop. This is not a collaborative multi-user editing protocol.

Each transcript notification contains a **complete segment snapshot**, not an append-only delta. The `version` field increments within a session and is not a global database revision number.

## 7. Persistence and exports

Storage directories are relative to the backend process working directory unless overridden:

| Default file | Data |
| --- | --- |
| `meetings/<meeting_id>/meeting.pkl` | `{"meta": ..., "state": ...}` |
| `speakers/speakers.pkl` | Speaker records, embeddings, and provenance |
| `hotwords/hotwords.pkl` | Named terminology libraries and timestamps |

Meeting metadata includes ID, title, created/updated timestamps, sample rate, merge gap, segment count, duration, and speaker count. The duration is based on the latest confirmed segment end, not a complete retained audio recording.

Resumable state includes participant IDs, hotword library IDs, thresholds, speaker-label mappings, next segment ID, and per-segment fields such as `id`, `start_sample`, `end_sample`, `text`, `emb`, `cluster`, `speaker_id`, and `speaker_name`.

Internal timing is in samples; the frontend uses milliseconds. With the standard sample rate:

`milliseconds = sample_index * 1000 / 16000`

Meeting IDs, speaker record IDs, and vector IDs are generally UUID-style hex strings. Segment IDs and displayed speaker IDs are integers. Do not mix these identifier types.

Most updates use a temporary file and `os.replace()` for replacement. Storage modules may use in-process locks; they are not designed for shared-write multi-process deployment.

**Raw meeting audio is not persisted.** Recorded audio cannot be played back or re-transcribed from a saved meeting, and a manual save does not necessarily process the current unconfirmed audio tail. Finalization is responsible for attempting to process the tail.

Markdown export contains title, metadata, timestamps, grouped speaker turns, and transcript text. JSON export includes metadata and transcript paragraphs, but not a full resumable session state or embeddings. Neither format is a complete replacement for the internal pickle snapshot.

These pickle files are for trusted local application data only. Never unpickle files received from untrusted sources.

## 8. HTTP API contract

All business endpoints start with `/api`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Settings, model load state, session counts |
| GET | `/api/meetings` | List meetings |
| POST | `/api/meetings` | Create a meeting with `{title}` |
| GET | `/api/meetings/{meeting_id}` | Meeting metadata and settings |
| GET | `/api/meetings/{meeting_id}/segments` | Segment summaries without embeddings |
| POST | `/api/meetings/{meeting_id}/save` | Save the active session |
| DELETE | `/api/meetings/{meeting_id}` | Delete a meeting |
| GET | `/api/meetings/{meeting_id}/export?format=md` | Export Markdown or JSON |
| GET | `/api/speakers` | List speakers and vector metadata |
| POST | `/api/speakers` | Create a speaker with `{name}` |
| PATCH | `/api/speakers/{speaker_id}` | Rename a speaker |
| DELETE | `/api/speakers/{speaker_id}` | Delete speaker and vectors |
| POST | `/api/speakers/{speaker_id}/vectors/record` | Enroll raw PCM voiceprint |
| POST | `/api/speakers/{speaker_id}/vectors/meeting` | Enroll from `{meeting_id, segment_ids}` |
| DELETE | `/api/speakers/{speaker_id}/vectors/{vector_id}` | Remove a voiceprint |
| GET | `/api/hotwords` | List hotword libraries |
| POST | `/api/hotwords` | Create a library with `{name, words}` |
| PATCH | `/api/hotwords/{hotword_id}` | Update name or words |
| DELETE | `/api/hotwords/{hotword_id}` | Delete a library |

Meeting metadata GET does not return the full transcript. The meeting page typically restores confirmed segments through WebSocket.

## 9. WebSocket protocol

Endpoint: `/ws/audio`.

Supported query parameters: `session_id`, `meeting_id`, `title`, `vad_threshold`, `rec_threshold`, `participants`, `hotwords`, `spk_threshold`. Participant and hotword selections are comma-separated ID lists.

Processing-window thresholds are clamped to 0.5–30 seconds, confirmation thresholds to 0.2–30 seconds, and speaker matching thresholds to 0–1. Empty explicit selections clear previously selected participants or hotword libraries.

Client messages include raw binary PCM frames and the stop control:

```json
{"type":"stop"}
```

Server messages:

| Type | Main fields | Meaning |
| --- | --- | --- |
| `ready` | session ID, sample rate, thresholds, meeting ID, resumed flag, participant/hotword selections | Connection initialization |
| `transcript` | version, segments | Complete confirmed-segment snapshot |
| `state` | buffered_ms | Current untrimmed buffer duration |
| `closed` | No required extra fields | Finalization complete |
| `error` | message | Possible processor failure |

Illustrative transcript message:

```json
{"type":"transcript","version":1,"segments":[{"id":1,"speaker_id":1,"speaker_name":"Speaker 1","start_ms":0,"end_ms":3200,"text":"The meeting has started."}]}
```

The current frontend server-message union and dispatcher do not explicitly handle the backend `error` type. Future protocol changes should update both ends.

## 10. Configuration

The backend reads environment variables directly. No automatic `.env` loading was observed in the source.

| Variable | Default | Role |
| --- | --- | --- |
| `VAD_THRESHOLD_S` | `10.0` | Buffer duration to trigger ASR |
| `REC_THRESHOLD_S` | `2.0` | Trailing sentence confirmation delay |
| `MERGE_GAP_S` | `2.0` | Maximum paragraph merge gap |
| `FINAL_DISCARD_S` | `1.0` | Final unconfirmed buffer discard threshold |
| `MEETINGS_DIR` | `meetings` | Meeting persistence directory |
| `SPEAKERS_DIR` | `speakers` | Voiceprint persistence directory |
| `HOTWORDS_DIR` | `hotwords` | Hotword persistence directory |
| `CLUSTER_MERGE_THR` | `0.78` | Speaker cluster merge threshold |
| `MIN_SPEAKER_AUDIO_S` | `5.0` | Voiceprint sample warning threshold |
| `SPK_MATCH_THRESHOLD` | `0.3` | Known-participant similarity threshold |
| `ASR_MODEL` | `paraformer-zh` | ASR model identifier or path |
| `FSMN_VAD_MODEL` | `fsmn-vad` | VAD model identifier or path |
| `PUNC_MODEL` | `ct-punc-c` | Punctuation model identifier or path |
| `SPK_MODEL` | `cam++` | Speaker embedding model identifier or path |
| `MODELSCOPE_CACHE` | ModelScope user cache | Root used for cached model lookup |

`resolve_model()` accepts existing local paths, maps known model aliases to ModelScope repository identifiers, and consults the cache. The presence of a `models_cache/` directory alone does not prove that it will be used; inspect cache layout and environment variables.

The frontend optionally reads `VITE_WS_URL` to override the **complete WebSocket endpoint URL**. By default, it uses the current origin with `/ws/audio`. The HTTP base path is `/api`.

## 11. Local development and migration

Run from the repository root in separate terminals:

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

```powershell
pnpm install
pnpm dev
```

Vite proxies `/api` and `/ws` to the backend on port 8000. The health URL is `http://127.0.0.1:8000/api/health` and FastAPI serves developer API docs at `/docs`.

The former `backend/backend/` and `frontend/frontend/` source trees were flattened into the repository root. The old single backend `app/` package was split into feature-based modules. The backend startup target changed from `app.main:app` to `main:app`. The manual ASR script imports `resolve_model` from `core.config`.

**Ignored local data is not moved by Git.** Stop old servers before moving existing meetings, speakers, hotwords, model caches, or local tools from old directories. Do not overwrite existing data. Reinstall dependencies in the root as needed.

The project's tracked source UI is English-only. Previously saved user content in other languages remains untouched. Chinese model aliases and Chinese-language recognition support are intentionally preserved.

## 12. Testing

Basic backend regression checks:

```powershell
uv run python -m unittest discover -s tests
```

Frontend type checking and build:

```powershell
pnpm type-check
pnpm build
```

The manual `example/stream_test.py` reads an external WAV, downmixes/resamples it, sends PCM through WebSocket, and prints confirmed transcripts and paragraphs. It does not automatically verify meeting persistence. The manual `example/test_stt.py` directly checks the FunASR models. Sample recordings are not tracked.

A complete automated suite for model inference, reconnect/resume behavior, concurrent sessions, long recordings, and audio quality is not present. Source-level consistency checks should not be reported as successful runtime inference tests.

## 13. Known limitations and follow-up items

These observations come from source inspection, not necessarily reproduced failures:

1. **ASR text depends on embeddings.** If voiceprint extraction returns no embedding for a confirmed sentence, the current processor can skip that sentence's recognized text.
2. **Save acknowledgment is not fully reliable.** The processor's save-complete event can remain set after a prior save; exceptions may be logged without being surfaced to callers; export does not strictly check the requested save result.
3. **Reconnect races.** Replacing a meeting session signals the old processor to stop without necessarily awaiting its final save before restoring a new connection. The frontend may replace a WebSocket client without explicitly closing the old client.
4. **Long meetings increase computation and transport.** Re-clustering, participant matching, and full transcript snapshots scale with accumulated segments. Unconfirmed buffers may become large.
5. **Concurrent inference is serialized.** All sessions share the global inference lock. Current registries and pickle files are not suitable as shared state for arbitrary multi-worker processes.
6. **Error handling is incomplete.** Backend `error` messages are not explicitly modeled by the frontend.
7. **No production access controls.** CORS permits arbitrary origins, and the application has no authentication, role isolation, or business-level rate limiting.
8. **Sensitive data and pickle boundaries.** Saved transcripts, voiceprint embeddings, and hotwords may contain sensitive content. Untrusted pickle data must never be loaded.

This English-only source cleanup changes the repository's tracked interface and documentation text, not user-created data. Old saved speaker labels and meeting titles may remain in their original language until intentionally changed by the user.

## 14. Development entrypoints

| Feature | Main files |
| --- | --- |
| Recognition timing and trailing audio | `sessions/processor.py`, `audio/buffer.py`, `core/config.py` |
| ASR, punctuation, embeddings, clustering | `inference/models.py` |
| Known-participant matching and speaker stability | `sessions/processor.py` |
| Meeting persistence and export | `storage/meetings.py`, `api/meetings.py` |
| Voiceprint enrollment and speaker records | `api/speakers.py`, `storage/speakers.py`, `src/components/SpeakerVectorAdder.vue` |
| Hotword libraries | `storage/hotwords.py`, `api/hotwords.py`, `src/pages/HotwordsPage.vue` |
| PCM capture and transmission | `src/audio/recorder.ts`, `src/audio/pcm-processor.ts`, `src/api/ws.ts` |
| Meeting view and UI state | `src/pages/MeetingPage.vue`, `src/pages/HomePage.vue`, `src/assets/main.css` |
| HTTP and WebSocket protocol | `api/`, `src/api/`, `src/types.ts` |
| Future summaries or action extraction | New post-processing modules, not implemented yet |

When extending the project, preserve the distinction between confirmed and unconfirmed segments, speaker matching and unknown-speaker clustering, sample and millisecond time units, and full snapshots versus incremental messages.

## 15. Suggested context for coding assistants

This repository contains a Vue/FastAPI local-first meeting transcription system. It uses FunASR Paraformer, FSMN-VAD, punctuation restoration, CAM++ speaker embeddings, and clustering; confirmed segments persist without raw audio. The backend is modular under `api/`, `core/`, `audio/`, `inference/`, `sessions/`, and `storage/`. The frontend lives in `src/`. Preserve the existing HTTP and WebSocket contracts, local pickle data compatibility, and Chinese speech-recognition support. Write new source code, comments, UI labels, tests, and documentation in English. Do not claim runtime validation without executing the relevant tests.

# Stability workstream: phase 01

This phase prevents silent loss of confirmed meeting text and makes reconnects and shutdowns more deterministic. Code is organized in independently testable modules.

## Backend boundaries

- `sessions/save_queue.py`: Per-request acknowledgment tickets. An HTTP request succeeds only when the corresponding write completes successfully.
- `sessions/outbox.py`: Prevents unbounded blocked send coroutines when a client is slow or disconnected.
- `sessions/identity/speaker_fallback.py`: Validates embeddings and creates stable unidentified-speaker labels if speaker extraction fails.
- `sessions/processor.py`: Drives ASR and durable saves. Confirmed text survives speaker classification problems.
- `api/session_lifecycle.py`: Serializes new connections to one meeting and waits for the previous session's finalization.
- `api/socket_transport.py`: Validates PCM frames before buffering them.
- `api/websocket.py`: Applies handover rules and reports errors and final save status.
- `api/meetings.py`: Rejects stale exports and coordinates deletion with active writers.

## Frontend boundaries

- `src/api/ws.ts`: Owns WebSocket connections and exposes errors.
- `src/types.ts`: Types server failures and final persistence results.
- `src/pages/MeetingPage.vue`: Closes previous clients and displays failed saves.
- `src/speaker/clustering.ts`: Colors unidentified speaker IDs correctly.

## Test layout

Backend unit tests live in `tests/backend/unit/`, API/session tests in `tests/backend/integration/`, UI protocol tests in `tests/frontend/contracts/`, and repository checks in `tests/quality/`.

Existing backend interface and English-only tests are relocated without dropping assertions. CI runs Python unittest discovery, Vue type checking/build, and Node realtime protocol tests.

## Scope limits

The backend still uses local pickle snapshots. Unit tests mock model outputs instead of loading FunASR weights. The existing global inference lock, growing clustering workload, full transcript snapshots, authentication, and multi-worker deployment remain separate tasks.

# Meeting of Pioneers

Meeting of Pioneers is a local-first meeting transcription application built with Vue 3, TypeScript, FastAPI, and FunASR. Its source code, interface labels, documentation, default names, logging descriptions, and error messages are maintained in English. The recognition engine still supports Chinese-language speech, and existing user-provided data is never translated automatically.

## Capabilities

- Live microphone capture with 16 kHz mono PCM streaming over WebSocket.
- Speech recognition with voice activity detection, punctuation, and timestamps.
- Speaker diarization, voiceprint enrollment, and participant matching.
- Reusable ASR hotword libraries for specialized terminology.
- Meeting creation, saving, resuming, deleting, and Markdown/JSON export.

This application does not currently implement automatic meeting summaries, task extraction, manual transcript editing, raw audio playback, cloud sync, accounts, or production authentication.

## Repository structure

```text
meeting/
├── main.py                    FastAPI application entrypoint
├── api/                       HTTP and WebSocket routes
│   ├── health.py
│   ├── meetings.py
│   ├── speakers.py
│   ├── hotwords.py
│   └── websocket.py
├── core/                      Settings and runtime dependencies
├── audio/                     Backend audio buffer
├── inference/                 ASR and speaker models
├── sessions/                  Connection and transcription processing
├── storage/                   Meeting, speaker, and hotword persistence
├── example/                   Manual audio test scripts
├── tests/                     Basic backend contract tests
├── src/                       Vue application source
│   ├── pages/
│   ├── components/
│   ├── api/
│   ├── audio/
│   ├── router/
│   └── assets/
├── public/                    Frontend static assets
├── index.html                 Vite HTML entrypoint
├── vite.config.ts             Vite dev proxy
├── package.json               Frontend dependencies
├── pnpm-lock.yaml             Frontend lock file
├── pyproject.toml             Backend dependencies
├── uv.lock                    Backend lock file
└── PROJECT_CONTEXT.md         Detailed architecture reference
```

The frontend and backend use the same repository root; there are no separate `frontend/` or `backend/` working directories.

## Local development

Requirements: Python 3.12 or newer, uv, pnpm, and Node.js `^22.18.0 || >=24.12.0`.

Open two terminals in the repository root.

**Backend**

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend**

```powershell
pnpm install
pnpm dev
```

Open the address printed by Vite and grant microphone access. The development proxy forwards `/api` and `/ws` requests to port 8000. The backend health endpoint is `http://127.0.0.1:8000/api/health`; FastAPI documentation is available at `/docs`.

Useful validation commands:

```powershell
uv run python -m unittest discover -s tests
pnpm type-check
pnpm build
```

To verify that all tracked source and documentation text remains free of CJK characters, run:

```powershell
python -m unittest tests.test_english_only
```

The Python test suite checks basic routes and persistence, not live FunASR inference or concurrent streaming. Initial startup may require downloading models. Override model identifiers or local paths using `ASR_MODEL`, `FSMN_VAD_MODEL`, `PUNC_MODEL`, and `SPK_MODEL`.

## Local data and migration

The application creates `meetings/`, `speakers/`, and `hotwords/` relative to the backend working directory. Git excludes these private directories, downloaded models, virtual environments, dependencies, recordings, and logs.

If you previously used `backend/backend/`, stop the old server before moving your ignored local `meetings/`, `speakers/`, and `hotwords/` directories into the repository root. Inspect `MODELSCOPE_CACHE` before moving model caches. Do not overwrite saved data. Git changes only tracked files and does not automatically relocate these local directories.

Existing transcript text, speaker names, and hotwords may legitimately contain Chinese or other languages. Converting the source interface to English does not rewrite user data and does not disable Chinese speech recognition. The application does not save raw meeting recordings. JSON exports are human-readable transcripts, not full resumable session backups.

The manual stream test requires an audio sample not tracked in this repository:

```powershell
uv run python example/stream_test.py --wav path/to/sample.wav
```

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for protocol details, settings, storage formats, architecture, limitations, and outstanding issues.

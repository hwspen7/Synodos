# Windows one-click startup

The project-root `Start-Meeting.cmd` launches the local stack with the selected Saluki 27B IQ2-mix GGUF model. It does not change or delete project files, transcripts, or model weights.

## Run

Double-click `Start-Meeting.cmd` in the project root. On the first launch only, if no saved PostgreSQL credential or `DATABASE_URL` exists, enter the password for local PostgreSQL user `meeting`. The launcher encrypts it using the current Windows account and writes it outside the repository to `E:\Personal_Project\meeting_pg_credential.xml`.

The launcher:

1. Uses `E:\AI\Ollama\Models`, port `11435`, and the existing Saluki 27B model. An already-running service on this port is reused.
2. Verifies the PostgreSQL connection using the existing encrypted credential. It does not switch back to the legacy file store.
3. Restarts a stale backend on port `8000` after verifying its process or its Python server plus the exact Meeting API signature (`/openapi.json`). Unknown processes are never terminated automatically.
4. Starts the backend with Whisper `large-v3` on CUDA and enables the Saluki-based Intelligence API.
5. Starts a dedicated AI worker with the same database and model configuration.
6. Starts or reuses the Vite frontend on port `5173`, checks the API state, and opens the browser.

The launcher installs missing small optional PostgreSQL/Intelligence dependencies and frontend packages when needed. The existing `.venv` with Python 3.12 or newer, required speech packages, PostgreSQL server, Node.js, pnpm, and Ollama must already be installed. The first ASR warmup can take longer if model weights need to be downloaded.

## Preconditions and safety

- Confirm that `E:\AI\Ollama\Models` contains the Saluki model and that Ollama can list it on port `11435`.
- Confirm that PostgreSQL is running on `127.0.0.1:5432`, with database `meeting` and user `meeting`.
- Confirm that `.venv\Scripts\python.exe` exists in the repository root.
- A saved `DATABASE_URL` environment variable takes precedence over the encrypted credential file.
- The launcher never logs the database password or embeds it in command-line arguments.
- If port `8000` or `5173` is owned by another application, the launcher stops with a diagnostic rather than killing that process.
- The frontend's enabled indicator confirms backend configuration, not completed AI inference. Check a real analysis job for `completed` status.
- Leave the service PowerShell windows open while using the application. Closing them stops their corresponding services.

## Troubleshooting

If `/api/intelligence/capabilities` still reports `qwen3:8b` or `enabled: false`, another backend is still listening on port `8000`. Inspect its command line with `Get-NetTCPConnection -LocalPort 8000 -State Listen` and `Get-CimInstance Win32_Process -Filter "ProcessId=<pid>"`.

If the encrypted PostgreSQL credential is invalid, rename the local `meeting_pg_credential.xml` file and run the launcher again to save a new credential. Do not send credentials in issue reports, screenshots, or chat messages.

If an AI job stays `queued`, inspect the separate worker PowerShell window. If inference fails, check the Ollama window and that the model exists in `ollama list` when `OLLAMA_HOST=127.0.0.1:11435`.

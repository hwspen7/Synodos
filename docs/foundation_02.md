# Meeting foundation patch 02

## Scope

This patch introduces six infrastructure foundations: PostgreSQL meeting and
library storage, explicit session lifecycle, a bounded inference scheduler,
operational logs and metrics, validated local security settings, and fault tests.
It is intended for the `refactor/flatten-project-root` branch with stability
patch 01 already applied. Do not delete the legacy folders.

## Important deployment boundary

This version supports one FastAPI application worker. PostgreSQL transactions
coordinate database writers, but the active WebSocket session registry remains
in-process. Running several Uvicorn workers is **not** supported until active
session ownership is coordinated across instances. Never expose this server
on the public Internet without authentication, TLS, and further security review.

## 1. Set up PostgreSQL (optional Docker example)

Install PostgreSQL 17 or Docker Desktop. In PowerShell, from the repository root:

```powershell
$env:POSTGRES_PASSWORD = Read-Host 'PostgreSQL password'
docker compose -f deploy/compose.postgres.yml up -d
```

Choose a long unique password. In the URL, URL-encode any reserved characters
in the password. Never commit real credentials or put them into screenshots.

Install PostgreSQL Python support:

```powershell
uv sync --extra postgres
```

If uv is not on PATH, invoke its absolute uv.exe path. Run all commands from the
project root. No PostgreSQL installation is needed for the default legacy mode.

## 2. Migration: explicitly requested, never automatic

Back up `meetings/`, `speakers/`, and `hotwords/` before migration. Shut down the
backend and stop all recordings, then set DATABASE_URL in your terminal.

```powershell
$env:DATABASE_URL = 'postgresql://meeting:REPLACE_ME@127.0.0.1:5432/meeting'
.\.venv\Scripts\python.exe -m storage.migrate_to_postgres
.\.venv\Scripts\python.exe -m storage.migrate_to_postgres --apply
```

The no-flag command only counts legacy records; `--apply` performs migration.
Existing PostgreSQL IDs are skipped, not replaced. A migration failure during
an individual meeting save is rolled back as a transaction and can be retried. Library data is imported
only if the relevant library has not already been initialized in PostgreSQL.
The legacy pickle files are never deleted, rewritten, or moved.

## 3. Enable PostgreSQL

```powershell
$env:MEETING_STORAGE_BACKEND = 'postgres'
$env:DATABASE_URL = 'postgresql://meeting:REPLACE_ME@127.0.0.1:5432/meeting'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

The PostgreSQL database and schema must be reachable. On connection errors,
startup fails explicitly instead of silently falling back to pickle.
The connection pool requires `psycopg` and `psycopg_pool`.

## Environment controls

- `MEETING_STORAGE_BACKEND`: `legacy` (default) or `postgres`.
- `DATABASE_URL`: PostgreSQL DSN; required with the postgres backend.
- `DB_POOL_MAX`: maximum database connections, default 5.
- `INFERENCE_QUEUE_SIZE`: bounded pending inference jobs, default 32.
- `INFERENCE_TIMEOUT_S`: maximum waiting time for inference, default 180.
- `MAX_AUDIO_BUFFER_SECONDS`: per-session unconfirmed PCM ceiling, default 120.
- `MAX_WS_CONNECTIONS`: connection limit per application process, default 12.
- `APP_ALLOWED_ORIGINS`: comma-separated allowed browser origins. Defaults to
  `http://localhost:5173,http://127.0.0.1:5173`.
- `APP_ALLOW_NO_ORIGIN_WS`: default false. Set true only for trusted nonbrowser
  clients, and never when exposing this server to untrusted networks.
- `APP_LOG_FORMAT`: `json` (default) or `plain`.
- `APP_LOG_LEVEL`: default `INFO`.
- `APP_ENABLE_METRICS`: default false. Metrics are available at `/api/metrics`
  only when enabled and still must be protected by network access controls.

## Automated verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
pnpm build
node --test tests/frontend/contracts/test_realtime_client.test.mjs
```

Database integration tests require a disposable database and TEST_DATABASE_URL.
Do not point TEST_DATABASE_URL to your production or personal database.

## Limitations

- Inference timeouts cannot abort an already running CUDA kernel.
- The model scheduler serializes shared GPU operations by default.
- PostgreSQL meeting snapshots currently rewrite full JSONB session state at
  save boundaries. Large-meeting incremental persistence is a subsequent change.
- The process cannot recover audio not yet transcribed or saved before a crash.
- The current `speaker` and `hotword` libraries are each stored as one JSONB
  document with a row lock per mutation, not normalized vector storage.
- PostgreSQL is not a backup system. Schedule `pg_dump` and test restoration.
- The configuration defaults to legacy storage to protect existing data until
  migration is explicitly performed and verified.

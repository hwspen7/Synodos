# Backend package layout

Source modules should live in a responsibility-based package directly under
the repository root. Avoid a dedicated top-level package for one generic module,
but do not merge unrelated product systems into one directory.

| Package | Purpose |
| --- | --- |
| `api/` | HTTP and WebSocket routes |
| `core/` | Runtime configuration, security, validation, metrics, and logging |
| `inference/` | Model manager and inference scheduling |
| `sessions/` | Meeting session lifecycle, PCM buffering, and speaker attribution |
| `speech/` | Speech recognition, alignment, diarization, and audio processing |
| `storage/` | Meeting and library persistence, PostgreSQL adapters |
| `intelligence/` | Summaries, retrieval, context corrections, background jobs |
| `runtime_quality/` | Phase-specific acceptance and recovery compatibility |
| `ops/` | Operational and backup scripts |
| `tools/` | Benchmarks, evaluations, and maintenance utilities |
| `tests/` | Subsystem-focused regression tests |

## Consolidated packages

- `audio/buffer.py` moved to `sessions/audio_buffer.py`.
- `foundation/settings.py` moved to `core/infrastructure_settings.py`.
- `foundation/security.py` moved to `core/security.py`.
- `foundation/validation.py` moved to `core/validation.py`.
- `observability/metrics.py` moved to `core/metrics.py`.
- `observability/logging_setup.py` moved to `core/logging_setup.py`.

The old `audio/`, `foundation/`, and `observability/` top-level Python
packages were removed. Python imports, unit tests, and compilation scripts
now use the new canonical paths. Exported public classes and functions retain
their existing names and behavior.

Keep `inference/` separate: its model lifecycle and bounded scheduler form
a cohesive domain. Keep `deploy/` and `config/` despite a single file each
because these are named operational input locations, not Python packages.
The existing nested test categories and PostgreSQL migration scripts are also
preserved.

All Vue, TypeScript, frontend styles and public assets under `src/` and
`public/` remain unchanged. This rearrangement does not change the API,
speech algorithms, database schema, or frontend behavior.

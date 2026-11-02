# Operational safety tools

All tools in this directory are opt-in. No application startup logic is changed.

- `Save-DatabaseCredential.ps1`: create an encrypted, current-Windows-user credential outside Git.
- `Backup-Postgres.ps1`: verified consistent PostgreSQL custom-format dump and retention.
- `Test-Restore.ps1`: verify archive integrity or restore into `meeting_restore_verify`.
- `Register-DailyBackup.ps1`: optional scheduled backup for the logged-in user.
- `Git-Checkpoint.ps1`: inspect, stage safe source files, commit and optionally push.
- `Run-Soak.ps1`: synthetic PCM traffic to a separate instance on port 8001.

For full setup steps, see the patch's `README_PATCH.md` before running scripts.

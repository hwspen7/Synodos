param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BackupRoot = 'E:\Personal_Project\meeting_postgres_backups',
    [string]$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml'
)

$ErrorActionPreference = 'Stop'
$logs = Join-Path $BackupRoot 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logs "scheduled_$stamp.log"
try {
    & (Join-Path $PSScriptRoot 'Backup-Postgres.ps1') `
        -ProjectRoot $ProjectRoot -BackupRoot $BackupRoot -CredentialFile $CredentialFile `
        *> $logFile
    if (-not $?) { throw 'Backup script failed.' }
} catch {
    "ERROR: Scheduled backup failed. Check PostgreSQL server and credentials. $($_.Exception.GetType().Name)" | Add-Content -LiteralPath $logFile
    exit 1
}
exit 0

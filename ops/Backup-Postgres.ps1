param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml',
    [string]$BackupRoot = 'E:\Personal_Project\meeting_postgres_backups',
    [string]$PgBin = 'C:\Program Files\PostgreSQL\17\bin',
    [int]$RetentionDays = 30,
    [int]$KeepLatest = 7
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Project Python is missing: $python" }
if (-not (Test-Path -LiteralPath $CredentialFile -PathType Leaf)) {
    throw "Encrypted credential missing. Run ops\Save-DatabaseCredential.ps1 first."
}
$credential = Import-Clixml -LiteralPath $CredentialFile
if ($credential.UserName -ne 'meeting') { throw 'Backup credential must belong to role meeting.' }
$prior = [Environment]::GetEnvironmentVariable('PGPASSWORD', 'Process')
try {
    $env:PGPASSWORD = $credential.GetNetworkCredential().Password
    Push-Location $ProjectRoot
    try {
        & $python -m ops.backup_manager backup `
            --pg-bin $PgBin --backup-root $BackupRoot `
            --retention-days $RetentionDays --keep-latest $KeepLatest
        if ($LASTEXITCODE -ne 0) { throw "PostgreSQL backup failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $prior) {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    } else {
        $env:PGPASSWORD = $prior
    }
}
Write-Host 'POSTGRESQL BACKUP VERIFIED' -ForegroundColor Green

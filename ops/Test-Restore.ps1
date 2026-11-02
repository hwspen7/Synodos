param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory=$true)][string]$Archive,
    [switch]$FullRestore,
    [string]$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml',
    [string]$PgBin = 'C:\Program Files\PostgreSQL\17\bin'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Archive does not exist: $Archive" }
if (-not (Test-Path -LiteralPath $python)) { throw "Project Python is missing: $python" }

Push-Location $ProjectRoot
try {
    if (-not $FullRestore) {
        & $python -m ops.backup_manager verify --pg-bin $PgBin `
            --backup-root (Split-Path -Parent $Archive) --archive $Archive
    } else {
        if (-not (Test-Path -LiteralPath $CredentialFile -PathType Leaf)) {
            throw 'Encrypted credential file is missing.'
        }
        $credential = Import-Clixml -LiteralPath $CredentialFile
        if ($credential.UserName -ne 'meeting') { throw 'Invalid database user' }
        $prior = [Environment]::GetEnvironmentVariable('PGPASSWORD', 'Process')
        try {
            $env:PGPASSWORD = $credential.GetNetworkCredential().Password
            & $python -m ops.restore_probe --archive $Archive --pg-bin $PgBin `
                --database meeting_restore_verify --confirm-disposable-database
        } finally {
            if ($null -eq $prior) {
                Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
            } else {
                $env:PGPASSWORD = $prior
            }
        }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Backup or restore verification failed.' }
} finally {
    Pop-Location
}

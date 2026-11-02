param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BackupRoot = 'E:\Personal_Project\meeting_postgres_backups',
    [string]$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml',
    [string]$TaskName = 'Meeting PostgreSQL Daily Backup',
    [string]$At = '20:00',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'ops\Backup-Postgres.ps1'))) {
    throw 'Backup script missing from project.'
}
if (-not (Test-Path -LiteralPath $CredentialFile)) {
    throw 'Run ops\Save-DatabaseCredential.ps1 before scheduling.'
}
if ($At -notmatch '^([01]\d|2[0-3]):[0-5]\d$') { throw 'Use an HH:mm time (24-hour clock).' }
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw 'A task with this name exists; do not overwrite it automatically.'
}
$script = Join-Path $ProjectRoot 'ops\Backup-Postgres.ps1'
$runner = Join-Path $ProjectRoot 'ops\Run-ScheduledBackup.ps1'
$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -ProjectRoot "{1}" -BackupRoot "{2}" -CredentialFile "{3}"' -f $runner, $ProjectRoot, $BackupRoot, $CredentialFile
Write-Host "Task: $TaskName"
Write-Host "Schedule: daily at $At (only while your Windows account is logged in)"
Write-Host "Backup root: $BackupRoot"
if ($DryRun) { Write-Host 'DRY RUN PASSED. No scheduled task created.'; return }
$powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Verified PostgreSQL custom backups for the Meeting application.' | Out-Null
Write-Host 'DAILY BACKUP TASK REGISTERED' -ForegroundColor Green
Write-Host 'Run it once manually using Start-ScheduledTask -TaskName ...'

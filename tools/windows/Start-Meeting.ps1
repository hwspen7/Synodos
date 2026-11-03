# Windows one-click launcher for Meeting of Pioneers.
# Child processes read their own configuration; database passwords never appear in process arguments.
[CmdletBinding()]
param(
    [ValidateSet('launcher', 'ollama', 'backend', 'worker', 'frontend')]
    [string]$Role = 'launcher'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$CredentialFile = 'E:\Personal_Project\meeting_pg_credential.xml'
$OllamaModels = 'E:\AI\Ollama\Models'
$OllamaHost = '127.0.0.1:11435'
$OllamaApi = 'http://127.0.0.1:11435'
$Model = 'hf.co/ConwayResearch/Underdog-Saluki-27B-1.0:Underdog-Saluki-27B-1.0-IQ2-mix.gguf'

function Set-AiEnvironment {
    $env:MEETING_AI_ENABLED = '1'
    $env:MEETING_AI_PROVIDER = 'ollama'
    $env:MEETING_AI_ENDPOINT = $OllamaApi
    $env:MEETING_AI_TEXT_MODEL = $Model
    $env:MEETING_AI_MODEL_MEETING = $Model
    $env:MEETING_AI_MODEL_SUMMARY = $Model
    $env:MEETING_AI_MODEL_EXTRACT = $Model
}

function Set-BackendEnvironment {
    Set-AiEnvironment
    $env:MEETING_ASR_ENGINE = 'whisper'
    $env:MEETING_ASR_MODEL = 'large-v3'
    $env:MEETING_ASR_DEVICE = 'cuda'
    $env:MEETING_ASR_COMPUTE = 'float16'
    $env:MEETING_ASR_LANGUAGE = 'en'
    $env:MEETING_TRANSLATE_TO = 'en'
    $env:MEETING_PREWARM_ASR = '1'
    $env:MEETING_RECORD_AUDIO = '1'
    $env:APP_ENABLE_METRICS = '1'
}

function Set-DatabaseEnvironment {
    $env:MEETING_STORAGE_BACKEND = 'postgres'
    if (-not [string]::IsNullOrWhiteSpace($env:DATABASE_URL)) {
        return
    }
    if (-not (Test-Path -LiteralPath $CredentialFile)) {
        throw "Missing encrypted PostgreSQL credential: $CredentialFile"
    }
    $credential = Import-Clixml -LiteralPath $CredentialFile
    if (-not ($credential -is [System.Management.Automation.PSCredential])) {
        throw 'The PostgreSQL credential file is not a PSCredential export.'
    }
    $user = [uri]::EscapeDataString($credential.UserName)
    $pass = [uri]::EscapeDataString($credential.GetNetworkCredential().Password)
    $env:DATABASE_URL = 'postgresql://{0}:{1}@127.0.0.1:5432/meeting' -f $user, $pass
    Remove-Variable credential, pass -ErrorAction SilentlyContinue
}

function Get-ListeningPid([int]$Port) {
    $entry = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $entry) { return $null }
    return [int]$entry.OwningProcess
}

function Get-ProcessInfo([int]$ProcessId) {
    return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
}

function Assert-ProjectProcess([int]$ProcessId, [string]$Marker, [int]$Port) {
    $processInfo = Get-ProcessInfo $ProcessId
    $command = if ($null -eq $processInfo) { '' } else { [string]$processInfo.CommandLine }
    if ($command.IndexOf($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
        $command.IndexOf($Marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw "Port $Port belongs to process $ProcessId, which is not an identified Meeting process. Close it manually."
    }
}

function Wait-For([scriptblock]$Check, [int]$TimeoutSeconds, [string]$Description) {
    $elapsed = 0
    while ($elapsed -le $TimeoutSeconds) {
        try {
            if (& $Check) { return $true }
        } catch {
            # An endpoint can be temporarily unavailable during startup.
        }
        if ($elapsed -ge $TimeoutSeconds) { break }
        if ($elapsed -gt 0 -and ($elapsed % 20) -eq 0) {
            Write-Host "Waiting for $Description ($elapsed seconds)..."
        }
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
    return $false
}

function Start-RoleWindow([string]$TargetRole) {
    $arguments = '-NoExit -NoProfile -ExecutionPolicy Bypass -File "{0}" -Role {1}' -f $PSCommandPath, $TargetRole
    Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $RepoRoot | Out-Null
    Write-Host "Started $TargetRole window."
}

function Get-AiCapabilities {
    return Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/intelligence/capabilities' -TimeoutSec 3
}

function Test-BackendReady {
    $ai = Get-AiCapabilities
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3
    return ($ai.enabled -eq $true -and $ai.text_model -eq $Model -and
        $health.infrastructure.database_backend -eq 'postgres')
}

function Test-OllamaReady {
    $data = Invoke-RestMethod -Uri "$OllamaApi/api/tags" -TimeoutSec 3
    foreach ($item in @($data.models)) {
        if ($item.name -eq $Model -or $item.model -eq $Model) { return $true }
    }
    return $false
}

function Test-DatabaseConnection {
    & $Python -c "import os, psycopg; psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=5).close()"
    return ($LASTEXITCODE -eq 0)
}

function Initialize-Credential {
    if (-not [string]::IsNullOrWhiteSpace($env:DATABASE_URL)) { return }
    if (Test-Path -LiteralPath $CredentialFile) { return }
    Write-Host 'First launch: PostgreSQL password is required once.'
    $secret = Read-Host -Prompt 'Password for local PostgreSQL user meeting' -AsSecureString
    if ($secret.Length -eq 0) { throw 'Database password cannot be empty.' }
    $credential = [System.Management.Automation.PSCredential]::new('meeting', $secret)
    $credential | Export-Clixml -LiteralPath $CredentialFile
    Remove-Variable credential, secret -ErrorAction SilentlyContinue
    Write-Host "Encrypted credential saved for this Windows account: $CredentialFile"
}

function Start-Launcher {
    Write-Host ''
    Write-Host 'Meeting of Pioneers - one-click startup' -ForegroundColor Cyan
    Write-Host 'AI: Saluki 27B  |  ASR: Whisper large-v3  |  DB: PostgreSQL'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python environment is missing: $Python"
    }
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        throw 'pnpm is not available. Install Node.js and pnpm before using this launcher.'
    }
    if (-not (Test-Path -LiteralPath $OllamaModels)) {
        throw "Ollama model directory is missing: $OllamaModels"
    }
    Set-Location -LiteralPath $RepoRoot

    & $Python -c 'import psycopg, psycopg_pool, uvicorn'
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing missing PostgreSQL and AI dependencies...'
        & $Python -m pip install 'psycopg[binary,pool]>=3.2,<4' -r (Join-Path $RepoRoot 'requirements-intelligence.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'node_modules\.bin\vite.cmd'))) {
        Write-Host 'Installing frontend dependencies for first launch...'
        & pnpm install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }

    $originalDatabaseUrl = $env:DATABASE_URL
    Initialize-Credential
    Set-DatabaseEnvironment
    if (-not (Get-ListeningPid 5432)) {
        $postgresServices = @(Get-Service -Name 'postgresql*' -ErrorAction SilentlyContinue)
        if ($postgresServices.Count -eq 1 -and $postgresServices[0].Status -ne 'Running') {
            Write-Host 'Starting the installed PostgreSQL Windows service...'
            try { Start-Service -Name $postgresServices[0].Name }
            catch { throw 'PostgreSQL service is stopped and could not be started. Start it from Windows Services.' }
        }
    }
    if (-not (Test-DatabaseConnection)) {
        throw "PostgreSQL connection failed. Check the running service and encrypted credential at $CredentialFile."
    }
    Write-Host 'PostgreSQL connection verified.' -ForegroundColor Green
    if ([string]::IsNullOrWhiteSpace($originalDatabaseUrl)) {
        Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
    } else {
        $env:DATABASE_URL = $originalDatabaseUrl
    }

    if (-not (Get-ListeningPid 11435)) {
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            throw 'Ollama executable not found.'
        }
        Start-RoleWindow 'ollama'
        if (-not (Wait-For { Test-OllamaReady } 40 'Ollama')) {
            throw 'Ollama did not start with the expected Saluki model. Inspect the Ollama window.'
        }
    }
    if (-not (Test-OllamaReady)) {
        throw 'Ollama on port 11435 is not exposing the expected Saluki model. Check OLLAMA_MODELS.'
    }
    Write-Host 'Saluki model found in Ollama.' -ForegroundColor Green

    $backendPid = Get-ListeningPid 8000
    $backendReady = $false
    if ($null -ne $backendPid) {
        try { $backendReady = Test-BackendReady } catch { $backendReady = $false }
        if (-not $backendReady) {
            Assert-ProjectProcess $backendPid 'uvicorn' 8000
            Assert-ProjectProcess $backendPid 'main:app' 8000
            Write-Host "Stopping stale Meeting backend (PID $backendPid)..."
            Stop-Process -Id $backendPid -Force
            if (-not (Wait-For { $null -eq (Get-ListeningPid 8000) } 12 'backend port release')) {
                throw 'Port 8000 is still in use; inspect the old backend process.'
            }
        }
    }
    if (-not $backendReady) {
        Start-RoleWindow 'backend'
        if (-not (Wait-For { Test-BackendReady } 300 'backend model initialization')) {
            throw 'Backend did not report Saluki + PostgreSQL on port 8000. Inspect the backend window.'
        }
    }
    Write-Host 'Backend AI configuration verified.' -ForegroundColor Green

    $workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object {
            $_.CommandLine -and $_.CommandLine.IndexOf($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $_.CommandLine.IndexOf('intelligence.worker', [StringComparison]::OrdinalIgnoreCase) -ge 0
        })
    $workerReusable = $false
    if ($workers.Count -eq 1) {
        $parentInfo = Get-ProcessInfo ([int]$workers[0].ParentProcessId)
        $parentCommand = if ($null -eq $parentInfo) { '' } else { [string]$parentInfo.CommandLine }
        $workerReusable = ($parentCommand.IndexOf('Start-Meeting.ps1', [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $parentCommand.IndexOf('-Role worker', [StringComparison]::OrdinalIgnoreCase) -ge 0)
    }
    if ($workerReusable) {
        Write-Host 'Reusing the AI worker started by this launcher.'
    } else {
        foreach ($existing in $workers) {
            Write-Host "Restarting existing project AI worker (PID $($existing.ProcessId))..."
            Stop-Process -Id $existing.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-RoleWindow 'worker'
        if (-not (Wait-For {
            @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
                $_.CommandLine -and $_.CommandLine.IndexOf($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $_.CommandLine.IndexOf('intelligence.worker', [StringComparison]::OrdinalIgnoreCase) -ge 0
            }).Count -gt 0
        } 20 'AI worker')) {
            throw 'AI worker did not start. Inspect the worker window.'
        }
    }
    Write-Host 'AI worker process detected.' -ForegroundColor Green

    $frontendPid = Get-ListeningPid 5173
    if ($null -ne $frontendPid) {
        Assert-ProjectProcess $frontendPid 'vite' 5173
        Write-Host 'Reusing existing Meeting frontend.'
    } else {
        Start-RoleWindow 'frontend'
    }
    if (-not (Wait-For {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -TimeoutSec 3 -UseBasicParsing
        $response.StatusCode -eq 200 -and $response.Content -like '*Meeting of Pioneers*'
    } 50 'frontend')) {
        throw 'Frontend did not start on port 5173. Inspect the frontend window.'
    }

    Write-Host ''
    Write-Host 'All services are available. Opening http://127.0.0.1:5173' -ForegroundColor Green
    Start-Process 'http://127.0.0.1:5173'
}

try {
    Set-Location -LiteralPath $RepoRoot
    switch ($Role) {
        'launcher' { Start-Launcher }
        'ollama' {
            $Host.UI.RawUI.WindowTitle = 'Meeting - Ollama Saluki 27B'
            $env:OLLAMA_MODELS = $OllamaModels
            $env:OLLAMA_HOST = $OllamaHost
            & ollama serve
        }
        'backend' {
            $Host.UI.RawUI.WindowTitle = 'Meeting - Backend'
            Set-BackendEnvironment
            Set-DatabaseEnvironment
            & $Python -m uvicorn main:app --host 127.0.0.1 --port 8000
        }
        'worker' {
            $Host.UI.RawUI.WindowTitle = 'Meeting - AI Worker'
            Set-AiEnvironment
            Set-DatabaseEnvironment
            & $Python -m intelligence.worker
        }
        'frontend' {
            $Host.UI.RawUI.WindowTitle = 'Meeting - Frontend'
            & pnpm dev --host 127.0.0.1 --port 5173 --strictPort
        }
    }
} catch {
    Write-Host "Startup error: $($_.Exception.Message)" -ForegroundColor Red
    throw
}

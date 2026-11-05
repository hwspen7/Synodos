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
    # Select tested ASR hybrid V2, but allow explicit Whisper rollback.
    if ([string]::IsNullOrWhiteSpace($env:MEETING_ASR_ENGINE)) {
        $env:MEETING_ASR_ENGINE = 'hybrid'
    }
    if ($env:MEETING_ASR_ENGINE -notin @('hybrid', 'whisper')) {
        throw "Launcher supports MEETING_ASR_ENGINE=hybrid or whisper"
    }
    if ($env:MEETING_ASR_ENGINE -eq 'hybrid') {
        $env:MEETING_HYBRID_QWEN_PYTHON = 'E:\AI-Cache\Environments\meeting-qwen-asr\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $env:MEETING_HYBRID_QWEN_PYTHON -PathType Leaf)) {
            throw "Hybrid ASR requires the E-drive Qwen Python environment: $env:MEETING_HYBRID_QWEN_PYTHON"
        }
    }
    $env:HF_HOME = 'E:\AI-Cache\MeetingReference\ASRSelection\cache\huggingface'
    $env:MEETING_ASR_MODEL = 'large-v3'
    $env:MEETING_ASR_DEVICE = 'cuda'
    $env:MEETING_ASR_COMPUTE = 'float16'
    $env:MEETING_ASR_LANGUAGE = 'auto'
    $env:MEETING_TRANSLATE_TO = 'en'
    $env:MEETING_PREWARM_ASR = '1'
    # Prefer CUDA DLLs bundled with the installed PyTorch wheel.
    # Windows CTranslate2 may not find them unless their folder is on PATH.
    $torchCudaLib = Join-Path $RepoRoot '.venv\Lib\site-packages\torch\lib'
    if (Test-Path -LiteralPath (Join-Path $torchCudaLib 'cublas64_12.dll')) {
        $env:PATH = "$torchCudaLib;$env:PATH"
        Write-Host "Configured CUDA 12 DLL search path: $torchCudaLib"
    } else {
        Write-Warning 'PyTorch CUDA 12 cuBLAS DLL not found in the local environment. Whisper GPU inference requires CUDA 12 cuBLAS and cuDNN 9.'
    }
    $env:MEETING_RECORD_AUDIO = '1'
    $env:APP_ENABLE_METRICS = '1'
    # Explicit real multi-speaker evaluation; never silently treat an ASR
    # single-speaker test as evidence of automatic diarization quality.
    # The Nemotron model is optional and remains off in ordinary launches.
    if ($env:MEETING_MULTISPEAKER_TEST -eq '1') {
        $env:MEETING_DIARIZATION_MODE = 'cluster'
        $env:MEETING_INDEPENDENT_DIARIZATION = 'nemotron3'
        $env:MEETING_CONTENT_FIRST_FUSION = '1'
        $env:MEETING_DIARIZATION_DIAGNOSTICS = '1'
        Write-Host 'Multi-speaker validation: independent Nemotron3 + CAM++ candidate enabled.'
    }
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

function Test-MeetingBackendIdentity {
    try {
        $schema = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/openapi.json' -TimeoutSec 3
        if ($schema.info.title -ne 'Meeting of Pioneers' -or $null -eq $schema.paths) {
            return $false
        }
        $paths = @($schema.paths.PSObject.Properties.Name)
        return ($paths -contains '/api/health' -and
            $paths -contains '/api/intelligence/capabilities')
    } catch {
        return $false
    }
}

function Test-MeetingFrontendIdentity {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -TimeoutSec 3 -UseBasicParsing
        return ($response.StatusCode -eq 200 -and
            $response.Content -like '*<title>Meeting of Pioneers</title>*' -and
            $response.Content -like '*/src/main.ts*')
    } catch {
        return $false
    }
}

function Assert-ProjectProcess([int]$ProcessId, [string]$Marker, [int]$Port) {
    $processInfo = Get-ProcessInfo $ProcessId
    if ($null -eq $processInfo) {
        throw "Port $Port owner PID $ProcessId could not be inspected. Stop it manually."
    }
    $command = [string]$processInfo.CommandLine
    $name = [string]$processInfo.Name
    if ($command.IndexOf($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $command.IndexOf($Marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return
    }

    # An earlier invocation may use a relative Python command without the
    # repository path. Confirm the exact application signature before stopping it.
    if ($Port -eq 8000 -and
        $name -match '^(python|pythonw|uvicorn)\.exe$' -and
        (Test-MeetingBackendIdentity)) {
        return
    }

    # Vite launched through pnpm can also have a pathless Node command line.
    # Reuse it only after verifying that this is the Meeting frontend.
    if ($Port -eq 5173 -and
        $name -match '^node\.exe$' -and
        (Test-MeetingFrontendIdentity)) {
        return
    }

    throw "Port $Port belongs to process $ProcessId ($name), which cannot be verified as Meeting. Close it manually."
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

function Test-BackendSourceUpdated([int]$ProcessId) {
    $processInfo = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $processInfo) { return $false }
    try {
        $started = $processInfo.StartTime.ToUniversalTime()
    } catch {
        return $false
    }
    foreach ($relativePath in @('main.py', 'api\websocket.py', 'api\intelligence.py',
                                  'intelligence\store.py', 'intelligence\contracts.py',
                                  'sessions\processor.py', 'speech\whisper.py',
                                  'speech\engine.py', 'speech\hybrid.py',
                                  'speech\qwen_bridge.py',
                                  'tools\windows\Start-Meeting.ps1',
                                  'core\runtime.py')) {
        $path = Join-Path $RepoRoot $relativePath
        if ((Test-Path -LiteralPath $path) -and
            (Get-Item -LiteralPath $path).LastWriteTimeUtc -gt $started.AddSeconds(1)) {
            return $true
        }
    }
    return $false
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
    Write-Host 'AI: Saluki 27B  |  ASR: Hybrid V2 (Whisper + Qwen 1.7B)  |  DB: PostgreSQL'
    if ($env:MEETING_MULTISPEAKER_TEST -eq '1') {
        Write-Host 'Multi-speaker test selected: force backend restart to activate a fresh Nemotron session.' -ForegroundColor Yellow
    }
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
        $sourceUpdated = Test-BackendSourceUpdated $backendPid
        $forceRestart = ($env:MEETING_FORCE_BACKEND_RESTART -eq '1' -or
            $env:MEETING_MULTISPEAKER_TEST -eq '1')
        if ($backendReady -and ($sourceUpdated -or $forceRestart)) {
            Write-Host 'Backend source code changed since startup; applying the updated version.'
            $backendReady = $false
        }
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
        $workerStart = (Get-Process -Id ([int]$workers[0].ProcessId)).StartTime.ToUniversalTime()
        foreach ($relPath in @('intelligence\worker.py', 'intelligence\context_correction.py',
                               'intelligence\contracts.py', 'intelligence\store.py')) {
            $workerSource = Join-Path $RepoRoot $relPath
            if ((Test-Path -LiteralPath $workerSource) -and
                (Get-Item -LiteralPath $workerSource).LastWriteTimeUtc -gt $workerStart.AddSeconds(1)) {
                $workerReusable = $false
                Write-Host 'AI worker source has changed. Restarting the previous worker.'
                break
            }
        }
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
            # The parent launcher may have checked before another Vite
            # instance bound its port. Re-check in the child process.
            $listeningPid = Get-ListeningPid 5173
            if ($null -ne $listeningPid) {
                Assert-ProjectProcess $listeningPid 'vite' 5173
                Write-Host "Meeting frontend already running (PID $listeningPid)."
            } else {
                # Keep one frontend start attempt at a time across multiple
                # simultaneous Start-Meeting.cmd windows.
                $mutex = New-Object System.Threading.Mutex(
                    $false, 'Local\MeetingOfPioneers_Vite_5173'
                )
                $entered = $false
                try {
                    $entered = $mutex.WaitOne(0)
                    if (-not $entered) {
                        if (-not (Wait-For {
                            $null -ne (Get-ListeningPid 5173)
                        } 30 'another frontend start')) {
                            throw 'Another Meeting frontend is initializing but did not bind port 5173.'
                        }
                        $listeningPid = Get-ListeningPid 5173
                        Assert-ProjectProcess $listeningPid 'vite' 5173
                        Write-Host "Reusing frontend started by another Meeting window (PID $listeningPid)."
                    } else {
                        # A second check closes the race between initial
                        # preflight and acquiring the single-start mutex.
                        $listeningPid = Get-ListeningPid 5173
                        if ($null -ne $listeningPid) {
                            Assert-ProjectProcess $listeningPid 'vite' 5173
                            Write-Host "Reusing existing Meeting frontend (PID $listeningPid)."
                        } else {
                            & pnpm dev --host 127.0.0.1 --port 5173 --strictPort
                            if ($LASTEXITCODE -ne 0) {
                                throw "Vite exited with code $LASTEXITCODE."
                            }
                        }
                    }
                } finally {
                    if ($entered) { [void]$mutex.ReleaseMutex() }
                    $mutex.Dispose()
                }
            }
        }
    }
} catch {
    Write-Host "Startup error: $($_.Exception.Message)" -ForegroundColor Red
    throw
}

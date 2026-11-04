# One-command, isolated Windows acceptance of independent Nemotron diarization.
# Never resets Git, deletes a benchmark, or stops a pre-existing web server.

[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [switch]$SkipUnitTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Port -lt 1024 -or $Port -gt 65535) {
    throw "Port must be 1024..65535."
}

if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repoRoot $PythonExe
}
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter not found: $PythonExe. Supply -PythonExe with your verified path."
}
$PythonExe = (Resolve-Path $PythonExe).Path

& $PythonExe -c "import sys; print('Interpreter:',sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Selected Python executable failed."
}

$root = "benchmark/fixed195/ami_shadow10"
$audio = Join-Path $root "audio/ami-TS3003b-1500-1740.wav"
$baseline = Join-Path $root "audit_TS3003b_v6_per_speaker.json"
if (-not (Test-Path $audio)) {
    throw "Frozen AMI audio missing: $audio"
}
if (-not (Test-Path $baseline)) {
    throw "Frozen V6 audit missing: $baseline"
}

if (-not $SkipUnitTests) {
    $modules = @(
        "tests.quality.test_nemotron_live_stream",
        "tests.quality.test_nemotron_processor_fusion",
        "tests.quality.test_ami_nemotron_live_score",
        "tests.quality.test_asr_stage_trace",
        "tests.quality.test_asr_priority_vs_nemotron"
    )
    foreach ($module in $modules) {
        & $PythonExe -m unittest $module -v
        if ($LASTEXITCODE -ne 0) {
            throw "Unit tests failed in $module. No server was started."
        }
    }
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$replay = Join-Path $root "replay_TS3003b_nemotron_$timestamp.json"
$score = Join-Path $root "audit_TS3003b_nemotron_$timestamp.json"
$gaps = Join-Path $root "audit_TS3003b_asr_stages_$timestamp.json"
$outLog = Join-Path $root "nemotron_server_$timestamp.stdout.log"
$errLog = Join-Path $root "nemotron_server_$timestamp.stderr.log"

# Set flags BEFORE spawning the private backend process. These are inherited
# only by that process and do not modify a different terminal's environment.
$env:MEETING_INDEPENDENT_DIARIZATION = "nemotron3"
$env:MEETING_NEMOTRON_STREAM_MODE = "low_latency"
$env:MEETING_NEMOTRON_DEVICE = "cuda"
$env:MEETING_DIARIZATION_DIAGNOSTICS = "1"
$env:MEETING_ASR_TRACE_STAGES = "1"
$env:MEETING_ASR_ENGINE = "whisper"
$env:MEETING_ASR_MODEL = "large-v3"
$env:MEETING_WHISPER_WINDOW_LID = "1"
$env:MEETING_WHISPER_SKIP_INTERNAL_VAD = "0"
$env:MEETING_SPEAKER_CLUSTER_STRATEGY = "bootstrap"
$env:MEETING_SPEAKER_PROBE_MIN_S = "1.2"
$env:MEETING_MIN_COMMIT_CONTEXT_S = "7.0"
$env:MEETING_ASR_SPEECH_BACKPRESSURE_GUARD = "0"
$env:MEETING_ASR_QUALITY_CANDIDATE = "0"
$env:MEETING_SPEAKER_QUALITY_CANDIDATE = "0"
$env:MEETING_WHISPER_GAP_RECOVERY = "0"
$env:MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT = "0"
$env:MEETING_ASR_COVERAGE_RETENTION = "0"
$env:MEETING_WHISPER_DUAL_VAD = "0"

# Preflight model weights, Transformers API and CUDA compatibility before
# starting a timed microphone replay. The subprocess caches weights locally.
& $PythonExe -c "from speech.nemotron_stream import _model_resources; _model_resources('low_latency'); print('Nemotron model preflight passed')"
if ($LASTEXITCODE -ne 0) {
    throw "Nemotron dependencies, CUDA or model weights are unavailable."
}

$healthUrl = "http://127.0.0.1:$Port/api/health"
try {
    $alreadyRunning = $null
    try {
        $alreadyRunning = Invoke-RestMethod $healthUrl -TimeoutSec 2
    } catch {
        $alreadyRunning = $null
    }
    if ($null -ne $alreadyRunning) {
        throw "Port $Port already hosts a service. Choose a different -Port; refusing to test an unknown backend."
    }

    $server = Start-Process -FilePath $PythonExe -WorkingDirectory $repoRoot `
        -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

    try {
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Seconds 2
            if ($server.HasExited) {
                throw "Backend exited during startup. Inspect $errLog"
            }
            try {
                $null = Invoke-RestMethod $healthUrl -TimeoutSec 2
                $ready = $true
                break
            } catch {
                # Cold model preload can extend startup time.
            }
        }
        if (-not $ready) {
            throw "Backend health endpoint did not become ready. Inspect $errLog"
        }

        $wsUrl = "ws://127.0.0.1:$Port/ws/audio?language=auto&record_audio=false&translate_to="
        & $PythonExe -m tools.live_caption_ws_replay `
            --audio $audio --loops 1 --speed 1.0 `
            --final-timeout 240 --url $wsUrl --output $replay
        if ($LASTEXITCODE -ne 0) {
            throw "Realtime replay failed. Inspect $errLog"
        }
        $r = Get-Content $replay -Raw | ConvertFrom-Json
        if ($r.server_ready.independent_diarization -ne "nemotron3") {
            throw "Wrong backend profile was used."
        }
        if ($r.delivery_matched -ne $true -or $r.server_closed -ne $true) {
            throw "Replay was not completed cleanly."
        }
        if ($r.independent_diarization.complete -ne $true) {
            throw "Nemotron failed: $($r.independent_diarization.error). Inspect $errLog"
        }
        if ($null -eq $r.final_cluster_diagnostics.anonymous_timeline) {
            throw "Missing caption-derived RTTM diagnostics."
        }

        & $PythonExe -m tools.ami_nemotron_live_score `
            --root $root --meeting TS3003b --replay $replay `
            --baseline $baseline --output $score --require-under-10
        $derExit = $LASTEXITCODE

        & $PythonExe -m tools.ami_asr_stage_gap_audit `
            --root $root --meeting TS3003b --replay $replay `
            --output $gaps
        if ($LASTEXITCODE -ne 0) {
            throw "ASR stage gap audit failed."
        }

        Write-Host ""
        Write-Host "Independent Nemotron report: $score"
        Write-Host "ASR missing-caption report: $gaps"
        Write-Host "Backend logs: $errLog"
        if ($derExit -ne 0) {
            throw "Frozen standard DER remains at or above 10%; NOT accepted."
        }
        Write-Host "Standard DER under 10% in this one frozen AMI sample."
        Write-Host "Further multi-meeting, multilingual, text-accuracy and latency validation is required."
    } finally {
        if ($null -ne $server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    Write-Error $_
    exit 1
}

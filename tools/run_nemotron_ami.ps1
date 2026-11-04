# One-command, isolated Windows acceptance of independent Nemotron diarization.
# Never resets Git, deletes a benchmark, or stops a pre-existing web server.

[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [switch]$SkipUnitTests,
    [switch]$WordBounds,
    [switch]$BypassWindowVad,
    [switch]$RecoverVadGaps,
    [switch]$RequirePublishedForcedUnder10
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Preserve one-variable-at-a-time attribution for ASR changes.
$candidates = @(
    $WordBounds.IsPresent,
    $BypassWindowVad.IsPresent,
    $RecoverVadGaps.IsPresent
) | Where-Object { $_ }
if (@($candidates).Count -gt 1) {
    throw "Choose only ONE ASR candidate: -WordBounds, -BypassWindowVad, or -RecoverVadGaps."
}

# Preserve existing E-drive model caches; redirect any unspecified/C-drive
# writable caches and temporary install paths into this E-drive worktree.
function Set-LocalCache([string]$Name, [string]$Relative) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $current -or $current -notmatch '^[Ee]:\\') {
        $current = Join-Path $repoRoot ("benchmark\\fixed195\\.runtime_cache\\" + $Relative)
        [Environment]::SetEnvironmentVariable($Name, $current, "Process")
    }
    New-Item -ItemType Directory -Path $current -Force | Out-Null
}
Set-LocalCache "HF_HOME" "huggingface"
Set-LocalCache "HF_HUB_CACHE" "huggingface\\hub"
Set-LocalCache "HF_XET_CACHE" "huggingface\\xet"
Set-LocalCache "HF_ASSETS_CACHE" "huggingface\\assets"
Set-LocalCache "TORCH_HOME" "torch"
Set-LocalCache "XDG_CACHE_HOME" "xdg"
Set-LocalCache "PIP_CACHE_DIR" "pip"
Set-LocalCache "TEMP" "temp"
Set-LocalCache "TMP" "temp"
Set-LocalCache "TMPDIR" "temp"

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
        "tests.quality.test_asr_priority_vs_nemotron",
        "tests.quality.test_caption_word_bounds_and_official_turns",
        "tests.quality.test_ami_nemotron_reference_diagnostic",
        "tests.quality.test_whisper_window_vad_ablation"
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
$forced = Join-Path $root "audit_TS3003b_forced_comparison_$timestamp.json"
$forcedGaps = Join-Path $root "audit_TS3003b_forced_asr_stages_$timestamp.json"
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
$env:MEETING_WHISPER_GAP_RECOVERY = if ($RecoverVadGaps.IsPresent) { "1" } else { "0" }
$env:MEETING_WHISPER_WINDOW_VAD_BYPASS = if ($BypassWindowVad.IsPresent) { "1" } else { "0" }
$env:MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT = "0"
$env:MEETING_ASR_COVERAGE_RETENTION = "0"
$env:MEETING_WHISPER_DUAL_VAD = "0"
$env:MEETING_CAPTION_WORD_BOUNDS = if ($WordBounds.IsPresent) { "1" } else { "0" }

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

        # Always compute the paired forced-alignment diagnostic. The
        # strict only_words gate above is intentionally NOT rewritten.
        & $PythonExe -m tools.ami_nemotron_reference_diagnostic `
            --root $root --meeting TS3003b --replay $replay `
            --output $forced
        if ($LASTEXITCODE -ne 0) {
            throw "Forced-alignment comparison failed."
        }

        & $PythonExe -m tools.ami_asr_stage_gap_audit `
            --root $root --meeting TS3003b --replay $replay `
            --reference-protocol forced_aligned --output $forcedGaps
        if ($LASTEXITCODE -ne 0) {
            throw "Forced-alignment ASR-stage gap audit failed."
        }

        $d = Get-Content $forced -Raw | ConvertFrom-Json
        $g = Get-Content $forcedGaps -Raw | ConvertFrom-Json
        $independentForcedPass = (
            [double]$d.forced_aligned_independent_der -lt 0.10
        )
        $publishedForcedPass = (
            [double]$d.forced_aligned_caption_der -lt 0.10
        )
        $clockMasked = [double]$d.forced_aligned_caption_clock_masked_acoustic_der

        Write-Host ""
        Write-Host "Independent Nemotron report: $score"
        Write-Host "ASR missing-caption report: $gaps"
        Write-Host "Paired forced-reference comparison: $forced"
        Write-Host "Forced-reference ASR-stage gaps: $forcedGaps"
        Write-Host "Backend logs: $errLog"
        Write-Host ("Original only_words independent DER: {0:P3}" -f ([double]$d.original_only_words_independent_der))
        Write-Host ("Forced reference independent DER: {0:P3}" -f ([double]$d.forced_aligned_independent_der))
        Write-Host ("Forced reference caption-clock counterfactual DER: {0:P3}" -f $clockMasked)
        Write-Host ("Forced reference ACTUAL published-caption DER: {0:P3}" -f ([double]$d.forced_aligned_caption_der))
        Write-Host ("Forced reference published speaker count: {0}" -f ([int]$d.forced_aligned_caption_metrics.predicted_speaker_count))
        Write-Host ("Word-aligned caption bounds active: {0}" -f $WordBounds.IsPresent)
        Write-Host ("Inner Whisper window VAD bypassed: {0}" -f $BypassWindowVad.IsPresent)
        Write-Host ("Bounded VAD-gap ASR retries: {0}" -f $RecoverVadGaps.IsPresent)
        Write-Host ("Forced-reference no-caption clock seconds: {0}" -f $g.reference_clock_no_caption_seconds)
        Write-Host ("Forced-reference VAD-without-raw-model seconds: {0}" -f $g.gap_causes_seconds.vad_without_raw_model_timestamp)
        Write-Host ("ASR recovery attempted / confirmed: {0} / {1}" -f $g.recovery_counts.recovery_attempted, $g.recovery_counts.recovery_confirmed)

        if ($RequirePublishedForcedUnder10.IsPresent -and -not $publishedForcedPass) {
            throw "Forced-reference PUBLISHED caption DER remains >=10%; NOT accepted."
        }
        if ($derExit -ne 0) {
            throw "Frozen original only_words independent DER remains >=10%; NOT accepted. Forced-reference scores are separate diagnostics above."
        }
        Write-Host "Original only_words independent DER under 10% on this one frozen AMI sample."
        if ($independentForcedPass) {
            Write-Host "Forced-alignment independent DER also under 10% (different reference protocol)."
        }
        if (-not $publishedForcedPass) {
            Write-Host "Final published-caption DER is NOT below 10% on the forced reference."
        }
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

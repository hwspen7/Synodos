# Run the Phase 1 and Phase 2 release preflight on the Windows test PC.
# All evidence stays under the private E-drive workspace.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [switch]$HumanReferenceVerified,
    [string]$Fixed195Results = "",
    [switch]$FullEvaluation,
    [string]$PyannotePython = "",
    [string]$DiarizenPython = "",
    [string]$QwenPython = "",
    [string]$NemotronReplayManifest = "",
    [string]$QwenAudio = "",
    [string]$QwenSegmentAudit = "",
    [string]$Phase2Dev = "",
    [string]$Phase2Holdout = "",
    [string]$PyannoteModelPath = "",
    [string]$DiarizenModelPath = "",
    [string]$DiarizenEmbeddingModelPath = "",
    [string]$QwenModelPath = "",
    [ValidateSet("cpu", "cuda")][string]$ExperimentDevice = "cuda",
    [switch]$AllowModelDownload,
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Run the acceptance checks from the E-drive project worktree."
}
if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter not found: $PythonExe"
}
$PythonExe = (Resolve-Path $PythonExe).Path
if ($RunId -and $RunId -notmatch '^[a-zA-Z0-9_-]{1,64}
$cacheBase = Join-Path $repo "benchmark\fixed195\.runtime_cache"
foreach ($name in @("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
                    "TORCH_HOME", "MODELSCOPE_CACHE", "PIP_CACHE_DIR",
                    "TEMP", "TMP", "TMPDIR")) {
    $current = [Environment]::GetEnvironmentVariable($name, "Process")
    if (-not $current -or -not $current.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
        $value = Join-Path $cacheBase $name.ToLowerInvariant()
        New-Item -Path $value -ItemType Directory -Force | Out-Null
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Write-Host "Phase 2: deterministic identity and manual-review tests"
$modules = @(
    "tests.backend.unit.test_phase2_identity_matcher",
    "tests.backend.unit.test_phase2_identity_consensus",
    "tests.backend.integration.test_phase2_manual_review",
    "tests.multilingual.test_corrections",
    "tests.quality.test_ami_human_text_reference",
    "tests.quality.test_phase1b_diarization",
    "tests.quality.test_phase1b_qwen_shadow",
    "tests.quality.test_phase2_cross_session_acceptance"
)
& $PythonExe -m unittest @modules -v
if ($LASTEXITCODE -ne 0) {
    throw "Phase 2 regression tests FAILED. Phase 1 replay was not started."
}

if ($FullEvaluation.IsPresent) {
    if (-not $PyannotePython -or -not $DiarizenPython -or
        -not $QwenPython -or -not $NemotronReplayManifest -or
        -not $QwenAudio -or -not $QwenSegmentAudit -or
        -not $Phase2Dev -or -not $Phase2Holdout) {
        throw "FullEvaluation requires all three optional Python environments, a ten-meeting replay manifest, a Qwen audio/segment audit pair, and separate Phase 2 dev/holdout manifests."
    }
    Write-Host "Phase 1A: independent Whisper ablations"
    $ablation = @{
        PythonExe = $PythonExe
        Port = $Port
        RunId = $runId
    }
    if ($HumanReferenceVerified.IsPresent) {
        $ablation.HumanReferenceVerified = $true
    }
    & (Join-Path $PSScriptRoot "run_phase1a_ablation.ps1") @ablation

    Write-Host "Phase 1B: paired, independent open-source diarization"
    $comparison = @{
        PythonExe = $PythonExe
        PyannotePython = $PyannotePython
        DiarizenPython = $DiarizenPython
        NemotronReplayManifest = $NemotronReplayManifest
        Device = $ExperimentDevice
        RunId = $runId
    }
    if ($PyannoteModelPath) { $comparison.PyannoteModelPath = $PyannoteModelPath }
    if ($DiarizenModelPath) { $comparison.DiarizenModelPath = $DiarizenModelPath }
    if ($DiarizenEmbeddingModelPath) {
        $comparison.DiarizenEmbeddingModelPath = $DiarizenEmbeddingModelPath
    }
    if ($AllowModelDownload.IsPresent) { $comparison.AllowModelDownload = $true }
    & (Join-Path $PSScriptRoot "run_phase1b_compare.ps1") @comparison

    Write-Host "Phase 1B: private, non-mutating Qwen3-ASR gap hypotheses"
    $qwenArgs = @(
        "-m", "tools.phase1b_qwen_shadow",
        "--audio", $QwenAudio, "--segment-audit", $QwenSegmentAudit,
        "--infer", "--language", "auto",
        "--output", (Join-Path $repo ("benchmark\fixed195\ami_shadow10\phase1b_qwen_$runId.json"))
    )
    if ($QwenModelPath) { $qwenArgs += @("--model-id", $QwenModelPath) }
    if ($AllowModelDownload.IsPresent) { $qwenArgs += "--allow-model-download" }
    & $QwenPython @qwenArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen3-ASR shadow evaluation failed"
    }

    Write-Host "Phase 2: disjoint cross-session identity threshold evaluation"
    & $PythonExe -m tools.phase2_cross_session_acceptance `
        --dev $Phase2Dev --holdout $Phase2Holdout `
        --output (Join-Path $repo ("benchmark\fixed195\phase2_identity_$runId.json"))
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
        throw "Phase 2 cross-session identity score failed to execute"
    }
    if ($LASTEXITCODE -eq 2) {
        Write-Warning "Phase 2 holdout thresholds were not met. Evidence will be included in the strict release report."
    }
} else {
    Write-Host "Model-free gates will run; optional GPU and held-out experiments require -FullEvaluation."
}

Write-Host "Phase 1: frozen ten-language, reviewed AMI replay and hosted CI checks"
$args = @{
    PythonExe = $PythonExe
    Port = $Port
    RunId = $runId
}
if ($HumanReferenceVerified.IsPresent) {
    $args.HumanReferenceVerified = $true
}
if ($Fixed195Results) {
    $args.Fixed195Results = $Fixed195Results
}
$closureFailure = $null
try {
    & (Join-Path $PSScriptRoot "run_phase1_closure.ps1") @args
} catch {
    $closureFailure = $_.Exception.Message
    Write-Warning ("Phase 1 closure did not pass: " + $closureFailure)
}

if ($FullEvaluation.IsPresent) {
    # Only artifacts with the exact current run ID may be submitted to
    # the all-or-nothing Phase 3 gate. Never pick a prior successful report.
    $private = Join-Path $repo "benchmark\fixed195"
    $ami = Join-Path $private "ami_shadow10"
    $inputs = [ordered]@{
        "language" = Join-Path $private ("phase1_10_final_$runId.json")
        "human" = Join-Path $ami ("audit_TS3003b_phase1_human_text_$runId.json")
        "acoustic" = Join-Path $ami ("audit_TS3003b_nemotron_$runId.json")
        "comparison" = Join-Path $ami ("phase1b\$runId\paired_comparison.json")
        "qwen-shadow" = Join-Path $ami ("phase1b_qwen_$runId.json")
        "identity" = Join-Path $private ("phase2_identity_$runId.json")
        "ci" = Join-Path $private ("ci_diagnostic_$runId.json")
        "ablation" = Join-Path $ami ("phase1a_ablation_$runId.json")
    }
    foreach ($kind in $inputs.Keys) {
        if (-not (Test-Path -LiteralPath $inputs[$kind] -PathType Leaf)) {
            throw "Phase 3 evidence is incomplete; the $kind report from this exact run is missing: $($inputs[$kind])"
        }
    }
    $decision = Join-Path $private ("phase12_release_decision_$runId.json")
    $releaseArguments = @("-m", "tools.phase12_release_gate")
    foreach ($kind in $inputs.Keys) {
        $releaseArguments += @("--$kind", $inputs[$kind])
    }
    $releaseArguments += @("--source-commit", $sourceCommit, "--output", $decision)
    & $PythonExe @releaseArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 1/2 release remains BLOCKED. Review $decision before Phase 3."
    }
    if ($closureFailure) {
        throw "Source release closure failed despite composite evidence: $closureFailure"
    }
    Write-Host "All eight same-run evidence gates passed. Phase 3 can begin. Manual merge review remains necessary."
} elseif ($closureFailure) {
    throw "Phase 1 closure remains BLOCKED: $closureFailure"
} else {
    Write-Host "Phase 1 standard preflight passed. Full Phase 1B/Phase 2 release validation still requires -FullEvaluation."
}
) {
    throw "RunId must be a safe local artifact identifier."
}
$runId = if ($RunId) { $RunId } else {
    (Get-Date -Format "yyyyMMdd-HHmmssfff") + "-$PID"
}
$sourceCommit = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[a-f0-9]{40}
$cacheBase = Join-Path $repo "benchmark\fixed195\.runtime_cache"
foreach ($name in @("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
                    "TORCH_HOME", "MODELSCOPE_CACHE", "PIP_CACHE_DIR",
                    "TEMP", "TMP", "TMPDIR")) {
    $current = [Environment]::GetEnvironmentVariable($name, "Process")
    if (-not $current -or -not $current.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
        $value = Join-Path $cacheBase $name.ToLowerInvariant()
        New-Item -Path $value -ItemType Directory -Force | Out-Null
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Write-Host "Phase 2: deterministic identity and manual-review tests"
$modules = @(
    "tests.backend.unit.test_phase2_identity_matcher",
    "tests.backend.unit.test_phase2_identity_consensus",
    "tests.backend.integration.test_phase2_manual_review",
    "tests.multilingual.test_corrections",
    "tests.quality.test_ami_human_text_reference",
    "tests.quality.test_phase1b_diarization",
    "tests.quality.test_phase1b_qwen_shadow",
    "tests.quality.test_phase2_cross_session_acceptance"
)
& $PythonExe -m unittest @modules -v
if ($LASTEXITCODE -ne 0) {
    throw "Phase 2 regression tests FAILED. Phase 1 replay was not started."
}

if ($FullEvaluation.IsPresent) {
    if (-not $PyannotePython -or -not $DiarizenPython -or
        -not $QwenPython -or -not $NemotronReplayManifest -or
        -not $QwenAudio -or -not $QwenSegmentAudit -or
        -not $Phase2Dev -or -not $Phase2Holdout) {
        throw "FullEvaluation requires all three optional Python environments, a ten-meeting replay manifest, a Qwen audio/segment audit pair, and separate Phase 2 dev/holdout manifests."
    }
    Write-Host "Phase 1A: independent Whisper ablations"
    $ablation = @{
        PythonExe = $PythonExe
        Port = $Port
    }
    if ($HumanReferenceVerified.IsPresent) {
        $ablation.HumanReferenceVerified = $true
    }
    & (Join-Path $PSScriptRoot "run_phase1a_ablation.ps1") @ablation

    Write-Host "Phase 1B: paired, independent open-source diarization"
    $comparison = @{
        PythonExe = $PythonExe
        PyannotePython = $PyannotePython
        DiarizenPython = $DiarizenPython
        NemotronReplayManifest = $NemotronReplayManifest
        Device = $ExperimentDevice
    }
    if ($PyannoteModelPath) { $comparison.PyannoteModelPath = $PyannoteModelPath }
    if ($DiarizenModelPath) { $comparison.DiarizenModelPath = $DiarizenModelPath }
    if ($DiarizenEmbeddingModelPath) {
        $comparison.DiarizenEmbeddingModelPath = $DiarizenEmbeddingModelPath
    }
    if ($AllowModelDownload.IsPresent) { $comparison.AllowModelDownload = $true }
    & (Join-Path $PSScriptRoot "run_phase1b_compare.ps1") @comparison

    Write-Host "Phase 1B: private, non-mutating Qwen3-ASR gap hypotheses"
    $qwenArgs = @(
        "-m", "tools.phase1b_qwen_shadow",
        "--audio", $QwenAudio, "--segment-audit", $QwenSegmentAudit,
        "--infer", "--language", "auto",
        "--output", (Join-Path $repo ("benchmark\fixed195\ami_shadow10\phase1b_qwen_" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json"))
    )
    if ($QwenModelPath) { $qwenArgs += @("--model-id", $QwenModelPath) }
    if ($AllowModelDownload.IsPresent) { $qwenArgs += "--allow-model-download" }
    & $QwenPython @qwenArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen3-ASR shadow evaluation failed"
    }

    Write-Host "Phase 2: disjoint cross-session identity threshold evaluation"
    & $PythonExe -m tools.phase2_cross_session_acceptance `
        --dev $Phase2Dev --holdout $Phase2Holdout `
        --output (Join-Path $repo ("benchmark\fixed195\phase2_identity_" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json"))
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2 cross-session identity threshold gate failed"
    }
} else {
    Write-Host "Model-free gates will run; optional GPU and held-out experiments require -FullEvaluation."
}

Write-Host "Phase 1: frozen ten-language, reviewed AMI replay and hosted CI checks"
$args = @{
    PythonExe = $PythonExe
    Port = $Port
}
if ($HumanReferenceVerified.IsPresent) {
    $args.HumanReferenceVerified = $true
}
if ($Fixed195Results) {
    $args.Fixed195Results = $Fixed195Results
}
& (Join-Path $PSScriptRoot "run_phase1_closure.ps1") @args
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1 closure runner failed."
}
Write-Host "Release preflight completed. Verify its private evidence files before entering Phase 3."
) {
    throw "Cannot pin a full Git commit for release evidence."
}
$cacheBase = Join-Path $repo "benchmark\fixed195\.runtime_cache"
foreach ($name in @("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
                    "TORCH_HOME", "MODELSCOPE_CACHE", "PIP_CACHE_DIR",
                    "TEMP", "TMP", "TMPDIR")) {
    $current = [Environment]::GetEnvironmentVariable($name, "Process")
    if (-not $current -or -not $current.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
        $value = Join-Path $cacheBase $name.ToLowerInvariant()
        New-Item -Path $value -ItemType Directory -Force | Out-Null
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Write-Host "Phase 2: deterministic identity and manual-review tests"
$modules = @(
    "tests.backend.unit.test_phase2_identity_matcher",
    "tests.backend.unit.test_phase2_identity_consensus",
    "tests.backend.integration.test_phase2_manual_review",
    "tests.multilingual.test_corrections",
    "tests.quality.test_ami_human_text_reference",
    "tests.quality.test_phase1b_diarization",
    "tests.quality.test_phase1b_qwen_shadow",
    "tests.quality.test_phase2_cross_session_acceptance"
)
& $PythonExe -m unittest @modules -v
if ($LASTEXITCODE -ne 0) {
    throw "Phase 2 regression tests FAILED. Phase 1 replay was not started."
}

if ($FullEvaluation.IsPresent) {
    if (-not $PyannotePython -or -not $DiarizenPython -or
        -not $QwenPython -or -not $NemotronReplayManifest -or
        -not $QwenAudio -or -not $QwenSegmentAudit -or
        -not $Phase2Dev -or -not $Phase2Holdout) {
        throw "FullEvaluation requires all three optional Python environments, a ten-meeting replay manifest, a Qwen audio/segment audit pair, and separate Phase 2 dev/holdout manifests."
    }
    Write-Host "Phase 1A: independent Whisper ablations"
    $ablation = @{
        PythonExe = $PythonExe
        Port = $Port
    }
    if ($HumanReferenceVerified.IsPresent) {
        $ablation.HumanReferenceVerified = $true
    }
    & (Join-Path $PSScriptRoot "run_phase1a_ablation.ps1") @ablation

    Write-Host "Phase 1B: paired, independent open-source diarization"
    $comparison = @{
        PythonExe = $PythonExe
        PyannotePython = $PyannotePython
        DiarizenPython = $DiarizenPython
        NemotronReplayManifest = $NemotronReplayManifest
        Device = $ExperimentDevice
    }
    if ($PyannoteModelPath) { $comparison.PyannoteModelPath = $PyannoteModelPath }
    if ($DiarizenModelPath) { $comparison.DiarizenModelPath = $DiarizenModelPath }
    if ($DiarizenEmbeddingModelPath) {
        $comparison.DiarizenEmbeddingModelPath = $DiarizenEmbeddingModelPath
    }
    if ($AllowModelDownload.IsPresent) { $comparison.AllowModelDownload = $true }
    & (Join-Path $PSScriptRoot "run_phase1b_compare.ps1") @comparison

    Write-Host "Phase 1B: private, non-mutating Qwen3-ASR gap hypotheses"
    $qwenArgs = @(
        "-m", "tools.phase1b_qwen_shadow",
        "--audio", $QwenAudio, "--segment-audit", $QwenSegmentAudit,
        "--infer", "--language", "auto",
        "--output", (Join-Path $repo ("benchmark\fixed195\ami_shadow10\phase1b_qwen_" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json"))
    )
    if ($QwenModelPath) { $qwenArgs += @("--model-id", $QwenModelPath) }
    if ($AllowModelDownload.IsPresent) { $qwenArgs += "--allow-model-download" }
    & $QwenPython @qwenArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen3-ASR shadow evaluation failed"
    }

    Write-Host "Phase 2: disjoint cross-session identity threshold evaluation"
    & $PythonExe -m tools.phase2_cross_session_acceptance `
        --dev $Phase2Dev --holdout $Phase2Holdout `
        --output (Join-Path $repo ("benchmark\fixed195\phase2_identity_" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json"))
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2 cross-session identity threshold gate failed"
    }
} else {
    Write-Host "Model-free gates will run; optional GPU and held-out experiments require -FullEvaluation."
}

Write-Host "Phase 1: frozen ten-language, reviewed AMI replay and hosted CI checks"
$args = @{
    PythonExe = $PythonExe
    Port = $Port
}
if ($HumanReferenceVerified.IsPresent) {
    $args.HumanReferenceVerified = $true
}
if ($Fixed195Results) {
    $args.Fixed195Results = $Fixed195Results
}
& (Join-Path $PSScriptRoot "run_phase1_closure.ps1") @args
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1 closure runner failed."
}
Write-Host "Release preflight completed. Verify its private evidence files before entering Phase 3."

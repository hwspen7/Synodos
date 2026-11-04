# Reproducible final Phase 1 evidence collection on the user's E-drive.
# Never merges, tags, rewrites human references or claims unreviewed overlap
# can be independently transcribed from mixed audio.

[CmdletBinding()]
param(
    [string]$PythonExe = ".\\.venv\\Scripts\\python.exe",
    [int]$Port = 8012,
    [switch]$HumanReferenceVerified,
    [switch]$SkipLiveReplay,
    [string]$Fixed195Results = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Phase 1 field acceptance must run from the E-drive worktree"
}
if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter is missing: $PythonExe"
}
$PythonExe = (Resolve-Path $PythonExe).Path

$root = Join-Path $repo "benchmark\\fixed195\\ami_shadow10"
$reference = Join-Path $root "TS3003b_1500_1740_official_orthographic.txt"
$metadata = Join-Path $root "TS3003b_1500_1740_official_orthographic.json"
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$summary = Join-Path $root "phase1_closure_evidence_$tag.json"
New-Item -ItemType Directory -Path $root -Force | Out-Null

$state = [ordered]@{
    schema = "meeting-phase1-closure-evidence-v1"
    source_commit = (& git rev-parse HEAD).Trim()
    human_reference_generated = $false
    human_reference_manual_review = $HumanReferenceVerified.IsPresent
    live_replay = "NOT_RUN"
    ci = "NOT_RUN"
    ten_language_test = "NOT_RUN"
    reference_path = $reference
    metadata_path = $metadata
    git_ci_diagnostic_path = $null
    safety_note = "Neither reference generation nor a CI diagnostic alone is a release acceptance result."
}

try {
    # Never overwrite previously generated human reference files.
    if (-not (Test-Path $reference) -or -not (Test-Path $metadata)) {
        if ((Test-Path $reference) -xor (Test-Path $metadata)) {
            throw "Incomplete reference pair; inspect existing file before proceeding"
        }
        $referenceArgs = @(
            "-m", "tools.ami_human_text_reference",
            "--root", "benchmark/fixed195/ami_shadow10",
            "--download-if-needed"
        )
        & $PythonExe @referenceArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Official AMI manual annotation extraction failed"
        }
    }
    $state.human_reference_generated = $true
    if (-not $Fixed195Results) {
        $candidates = @(
            Get-ChildItem "benchmark/fixed195" -Filter "content_review_singles_*.json" -File |
            Where-Object { $_.Name -notmatch '(_summary|_triage)\.json
        $report = Join-Path $repo "benchmark\\fixed195\\phase1_10_final_$tag.json"
        $gateArgs = @(
            "-m", "tools.phase1_ten_language_gate",
            "--root", "benchmark/fixed195",
            "--results", $Fixed195Results, "--output", $report
        )
        & $PythonExe @gateArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Ten-language reference-backed benchmark not accepted"
        }
        $state.ten_language_test = "PASS"
    }
    if (-not $SkipLiveReplay.IsPresent) {
        $sha = (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash.ToLowerInvariant()
        $replayArgs = @{
            PythonExe = $PythonExe
            Port = $Port
            HumanReferenceText = $reference
            ExpectedReferenceSha256 = $sha
            SkipUnitTests = $true
            ContentFirst = $true
            ExportPrivateTranscript = $true
        }
        if ($HumanReferenceVerified.IsPresent) {
            $replayArgs.HumanReferenceVerified = $true
        } else {
            Write-Warning "Human AMI words are not yet confirmed audible in the mixed audio; the strict text gate will fail closed."
        }
        try {
            & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @replayArgs
            $state.live_replay = "SCRIPT_PASSED_ALL_GATES"
        } catch {
            $state.live_replay = "NOT_ACCEPTED"
            Write-Warning ("Real meeting replay did not pass release gates: " + $_.Exception.Message)
            Write-Host "Inspect private TS3003b transcript and human audit under $root"
        }
    }
} catch {
    $state.live_replay = "ERROR_BEFORE_OR_DURING_REPLAY"
    Write-Warning ("AMI acceptance interrupted: " + $_.Exception.Message)
}

$ciFile = Join-Path $repo "benchmark\\fixed195\\ci_diagnostic_$tag.json"
$state.git_ci_diagnostic_path = $ciFile
try {
    $ciArgs = @{
        Commit = $state.source_commit
        Output = $ciFile
    }
    & (Join-Path $PSScriptRoot "diagnose_github_actions.ps1") @ciArgs
    $state.ci = "PASS"
} catch {
    $state.ci = "NOT_ACCEPTED"
    Write-Warning ("GitHub CI is not yet accepted: " + $_.Exception.Message)
}

$state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summary -Encoding UTF8
Write-Host "Final evidence summary: $summary"
Write-Host ("Live meeting status: {0}; GitHub CI status: {1}" -f $state.live_replay, $state.ci)
if ($state.live_replay -ne "SCRIPT_PASSED_ALL_GATES" -or $state.ci -ne "PASS" -or $state.ten_language_test -eq "NOT_RUN") {
    throw "Phase 1 release is BLOCKED until reference-backed text and hosted CI pass. Nothing was merged or tagged."
}
Write-Host "Local evidence checks passed. Merge/tag remains a separate controlled release operation."
 } |
            Sort-Object LastWriteTime -Descending
        )
        if ($candidates.Count -gt 0) {
            $Fixed195Results = $candidates[0].FullName
        } elseif (Test-Path "benchmark/fixed195/content_review_singles.json") {
            $Fixed195Results = "benchmark/fixed195/content_review_singles.json"
        }
    }
    if ($Fixed195Results) {
        $report = Join-Path $repo "benchmark\\fixed195\\phase1_10_final_$tag.json"
        $gateArgs = @(
            "-m", "tools.phase1_ten_language_gate",
            "--root", "benchmark/fixed195",
            "--results", $Fixed195Results, "--output", $report
        )
        & $PythonExe @gateArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Ten-language reference-backed benchmark not accepted"
        }
        $state.ten_language_test = "PASS"
    }
    if (-not $SkipLiveReplay.IsPresent) {
        $sha = (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash.ToLowerInvariant()
        $replayArgs = @{
            PythonExe = $PythonExe
            Port = $Port
            HumanReferenceText = $reference
            ExpectedReferenceSha256 = $sha
            SkipUnitTests = $true
            ContentFirst = $true
            ExportPrivateTranscript = $true
        }
        if ($HumanReferenceVerified.IsPresent) {
            $replayArgs.HumanReferenceVerified = $true
        } else {
            Write-Warning "Human AMI words are not yet confirmed audible in the mixed audio; the strict text gate will fail closed."
        }
        try {
            & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @replayArgs
            $state.live_replay = "SCRIPT_PASSED_ALL_GATES"
        } catch {
            $state.live_replay = "NOT_ACCEPTED"
            Write-Warning ("Real meeting replay did not pass release gates: " + $_.Exception.Message)
            Write-Host "Inspect private TS3003b transcript and human audit under $root"
        }
    }
} catch {
    $state.live_replay = "ERROR_BEFORE_OR_DURING_REPLAY"
    Write-Warning ("AMI acceptance interrupted: " + $_.Exception.Message)
}

$ciFile = Join-Path $repo "benchmark\\fixed195\\ci_diagnostic_$tag.json"
$state.git_ci_diagnostic_path = $ciFile
try {
    $ciArgs = @{
        Commit = $state.source_commit
        Output = $ciFile
    }
    & (Join-Path $PSScriptRoot "diagnose_github_actions.ps1") @ciArgs
    $state.ci = "PASS"
} catch {
    $state.ci = "NOT_ACCEPTED"
    Write-Warning ("GitHub CI is not yet accepted: " + $_.Exception.Message)
}

$state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summary -Encoding UTF8
Write-Host "Final evidence summary: $summary"
Write-Host ("Live meeting status: {0}; GitHub CI status: {1}" -f $state.live_replay, $state.ci)
if ($state.live_replay -ne "SCRIPT_PASSED_ALL_GATES" -or $state.ci -ne "PASS" -or $state.ten_language_test -eq "NOT_RUN") {
    throw "Phase 1 release is BLOCKED until reference-backed text and hosted CI pass. Nothing was merged or tagged."
}
Write-Host "Local evidence checks passed. Merge/tag remains a separate controlled release operation."

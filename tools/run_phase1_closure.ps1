# One-command, evidence-only Phase 1 release preflight for Windows E-drive.
# Never merges or tags; human-annotated overlap needs listening verification.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [switch]$HumanReferenceVerified,
    [switch]$SkipLiveReplay,
    [string]$Fixed195Results = "",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Run the Phase 1 preflight from the E-drive project worktree."
}
if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter not found: $PythonExe"
}
$PythonExe = (Resolve-Path $PythonExe).Path

$root = Join-Path $repo "benchmark\fixed195\ami_shadow10"
$reference = Join-Path $root "TS3003b_1500_1740_official_orthographic.txt"
$metadata = Join-Path $root "TS3003b_1500_1740_official_orthographic.json"
if ($RunId -and $RunId -notmatch '^[a-zA-Z0-9_-]{1,64}$') {
    throw "RunId must be a safe local artifact identifier."
}
$tag = if ($RunId) { $RunId } else { Get-Date -Format "yyyyMMdd-HHmmss" }
$summary = Join-Path $root "phase1_closure_evidence_$tag.json"
New-Item -ItemType Directory -Path $root -Force | Out-Null

$state = [ordered]@{
    schema = "meeting-phase1-closure-evidence-v1"
    source_commit = (& git rev-parse HEAD).Trim()
    source_reference = "AMI public manual v1.6.2, unreviewed mixed-audio words"
    human_reference_available = $false
    human_reference_manual_review = $HumanReferenceVerified.IsPresent
    live_replay = "NOT_RUN"
    ci = "NOT_RUN"
    ten_language_test = "NOT_RUN"
    reference_path = $reference
    metadata_path = $metadata
    ci_diagnostic_path = $null
}

try {
    if (-not (Test-Path $reference) -or -not (Test-Path $metadata)) {
        if ((Test-Path $reference) -xor (Test-Path $metadata)) {
            throw "Human reference metadata pair is incomplete; do not overwrite old data."
        }
        $refArgs = @(
            "-m", "tools.ami_human_text_reference",
            "--root", "benchmark/fixed195/ami_shadow10",
            "--download-if-needed"
        )
        & $PythonExe @refArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Official AMI annotation extraction failed."
        }
    }
    $state.human_reference_available = $true

    if (-not $Fixed195Results) {
        $sourceFiles = @(
            Get-ChildItem "benchmark/fixed195" -Filter "content_review_singles_*.json" -File |
                Where-Object { $_.Name -notmatch '(_summary|_triage)\.json$' } |
                Sort-Object LastWriteTime -Descending
        )
        if ($sourceFiles.Count -gt 0) {
            $Fixed195Results = $sourceFiles[0].FullName
        } elseif (Test-Path "benchmark/fixed195/content_review_singles.json") {
            $Fixed195Results = "benchmark/fixed195/content_review_singles.json"
        }
    }
    if ($Fixed195Results) {
        $gateOutput = Join-Path $repo "benchmark\fixed195\phase1_10_final_$tag.json"
        $gateArgs = @(
            "-m", "tools.phase1_ten_language_gate",
            "--root", "benchmark/fixed195",
            "--results", $Fixed195Results, "--output", $gateOutput
        )
        & $PythonExe @gateArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Ten-language ASR gate failed."
        }
        $state.ten_language_test = "PASS"
    } else {
        Write-Warning "No complete Fixed195 raw results found for rechecking."
    }

    if (-not $SkipLiveReplay.IsPresent) {
        $referenceSha = (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash.ToLowerInvariant()
        $replayArguments = @{
            PythonExe = $PythonExe
            Port = $Port
            HumanReferenceText = $reference
            ExpectedReferenceSha256 = $referenceSha
            SkipUnitTests = $true
            ContentFirst = $true
            ExportPrivateTranscript = $true
            OutputTag = $tag
        }
        if ($HumanReferenceVerified.IsPresent) {
            $replayArguments.HumanReferenceVerified = $true
        } else {
            Write-Warning "Official annotators transcribed overlapping voices, not necessarily all audible in a mixed microphone. Human review must precede acceptance."
        }
        try {
            & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @replayArguments
            $state.live_replay = "SCRIPT_PASSED_ALL_GATES"
        } catch {
            $state.live_replay = "NOT_ACCEPTED"
            Write-Warning ("Live text/diarization release gates are not passed: " + $_.Exception.Message)
            Write-Host "Inspect the private replay, transcript and text gate JSON in $root"
        }
    }
} catch {
    $state.live_replay = "ERROR_BEFORE_OR_DURING_REPLAY"
    Write-Warning ("AMI replay preparation failed: " + $_.Exception.Message)
}

$ciOutput = Join-Path $repo "benchmark\fixed195\ci_diagnostic_$tag.json"
$state.ci_diagnostic_path = $ciOutput
try {
    $ciArgs = @{ Commit = $state.source_commit; Output = $ciOutput }
    & (Join-Path $PSScriptRoot "diagnose_github_actions.ps1") @ciArgs
    $state.ci = "PASS"
} catch {
    $state.ci = "NOT_ACCEPTED"
    Write-Warning ("GitHub Actions still not accepted: " + $_.Exception.Message)
}

$state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summary -Encoding UTF8
Write-Host "Private evidence summary: $summary"
Write-Host ("Real meeting: {0} | Hosted CI: {1}" -f $state.live_replay, $state.ci)
if ($state.live_replay -ne "SCRIPT_PASSED_ALL_GATES" -or
    $state.ci -ne "PASS" -or
    $state.ten_language_test -ne "PASS") {
    throw "Phase 1 release is BLOCKED. No commit was merged or tagged."
}
Write-Host "Evidence gates passed; final merge/tag still requires release review."

# Run the existing bounded Whisper ablation modes on the SAME AMI audio.
# Results are experimental; no threshold is relaxed and no live ASR changes.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [string]$HumanReferenceText = "",
    [switch]$HumanReferenceVerified,
    [string]$RunId = ""
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Phase 1A experiments must run in the E-drive worktree."
}
if (-not [IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
$private = Join-Path $repo "benchmark\fixed195\ami_shadow10"
if (-not $HumanReferenceText) {
    $HumanReferenceText = Join-Path $private "TS3003b_1500_1740_official_orthographic.txt"
}
$HumanReferenceText = (Resolve-Path -LiteralPath $HumanReferenceText).Path
$sha = (Get-FileHash -LiteralPath $HumanReferenceText -Algorithm SHA256).Hash.ToLowerInvariant()
if ($RunId -and $RunId -notmatch '^[a-zA-Z0-9_-]{1,64}
$output = Join-Path $private ("phase1a_ablation_$runId.json")
$variants = @(
    @{ name = "content_first"; switch = "ContentFirst" },
    @{ name = "whisper_window_vad_bypass"; switch = "BypassWindowVad" },
    @{ name = "bounded_vad_gap_recovery"; switch = "RecoverVadGaps" },
    @{ name = "preserve_verified_unaligned"; switch = "PreserveVerifiedText" },
    @{ name = "word_timestamp_bounds"; switch = "WordBounds" }
)
$records = @()
foreach ($variant in $variants) {
    $before = @(Get-ChildItem -LiteralPath $private -Filter "audit_TS3003b_nemotron_*.json" -File -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $args = @{
        PythonExe = $PythonExe
        Port = $Port
        SkipUnitTests = $true
        ExportPrivateTranscript = $true
        HumanReferenceText = $HumanReferenceText
        ExpectedReferenceSha256 = $sha
    }
    $args[$variant.switch] = $true
    if ($HumanReferenceVerified.IsPresent) {
        $args.HumanReferenceVerified = $true
    }
    $state = "REPLAY_RAN_NOT_ACCEPTED"
    $errorText = ""
    Write-Host "Phase 1A ablation: $($variant.name)"
    try {
        & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @args
        $state = "SCRIPT_PASSED"
    } catch {
        $errorText = $_.Exception.Message
    }
    $after = @(Get-ChildItem -LiteralPath $private -Filter "audit_TS3003b_nemotron_*.json" -File -ErrorAction SilentlyContinue)
    $new = @($after | Where-Object { $before -notcontains $_.FullName } |
        Sort-Object LastWriteTime -Descending)
    $reportPath = if ($new.Count -gt 0) { $new[0].FullName } else { $null }
    $records += [pscustomobject]@{
        variant = $variant.name
        execution_status = $state
        independent_der_path = $reportPath
        error = $errorText
    }
}
$document = [ordered]@{
    schema = "meeting-phase1a-five-candidate-ablation-v1"
    source_commit = (& git rev-parse HEAD).Trim()
    audio_meeting = "TS3003b"
    reference_sha256 = $sha
    reference_human_verified = $HumanReferenceVerified.IsPresent
    candidate_trials = $records
    official_release_accepted = $false
    note = "Candidate reports require independent same-audio reference and human review. No model score is altered or auto-promoted."
}
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $output -Encoding UTF8
Write-Host "Private ablation inventory: $output"
$comparePath = Join-Path $private ("phase1a_comparison_$runId.json")
& $PythonExe -m tools.phase1a_compare --inventory $output --output $comparePath
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
    throw "Phase 1A comparison failed to verify paired audit evidence."
}
Write-Host "Private paired acoustic/text review comparison: $comparePath"
if (@($records | Where-Object { -not $_.independent_der_path }).Count -gt 0) {
    throw "Some Phase 1A ablations did not produce acoustic evidence."
}
Write-Host "Five Phase 1A candidate replays produced diagnostic evidence; review actual CER/DER before selecting."
) {
    throw "RunId must be safe for private filenames."
}
$runId = if ($RunId) { $RunId } else { (Get-Date -Format "yyyyMMdd-HHmmss") + "-$PID" }
$output = Join-Path $private ("phase1a_ablation_$runId.json")
$variants = @(
    @{ name = "content_first"; switch = "ContentFirst" },
    @{ name = "whisper_window_vad_bypass"; switch = "BypassWindowVad" },
    @{ name = "bounded_vad_gap_recovery"; switch = "RecoverVadGaps" },
    @{ name = "preserve_verified_unaligned"; switch = "PreserveVerifiedText" },
    @{ name = "word_timestamp_bounds"; switch = "WordBounds" }
)
$records = @()
foreach ($variant in $variants) {
    $before = @(Get-ChildItem -LiteralPath $private -Filter "audit_TS3003b_nemotron_*.json" -File -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $args = @{
        PythonExe = $PythonExe
        Port = $Port
        SkipUnitTests = $true
        ExportPrivateTranscript = $true
        HumanReferenceText = $HumanReferenceText
        ExpectedReferenceSha256 = $sha
    }
    $args[$variant.switch] = $true
    if ($HumanReferenceVerified.IsPresent) {
        $args.HumanReferenceVerified = $true
    }
    $state = "REPLAY_RAN_NOT_ACCEPTED"
    $errorText = ""
    Write-Host "Phase 1A ablation: $($variant.name)"
    try {
        & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @args
        $state = "SCRIPT_PASSED"
    } catch {
        $errorText = $_.Exception.Message
    }
    $after = @(Get-ChildItem -LiteralPath $private -Filter "audit_TS3003b_nemotron_*.json" -File -ErrorAction SilentlyContinue)
    $new = @($after | Where-Object { $before -notcontains $_.FullName } |
        Sort-Object LastWriteTime -Descending)
    $reportPath = if ($new.Count -gt 0) { $new[0].FullName } else { $null }
    $records += [pscustomobject]@{
        variant = $variant.name
        execution_status = $state
        independent_der_path = $reportPath
        error = $errorText
    }
}
$document = [ordered]@{
    schema = "meeting-phase1a-five-candidate-ablation-v1"
    source_commit = (& git rev-parse HEAD).Trim()
    audio_meeting = "TS3003b"
    reference_sha256 = $sha
    reference_human_verified = $HumanReferenceVerified.IsPresent
    candidate_trials = $records
    official_release_accepted = $false
    note = "Candidate reports require independent same-audio reference and human review. No model score is altered or auto-promoted."
}
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $output -Encoding UTF8
Write-Host "Private ablation inventory: $output"
$comparePath = Join-Path $private ("phase1a_comparison_$runId.json")
& $PythonExe -m tools.phase1a_compare --inventory $output --output $comparePath
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
    throw "Phase 1A comparison failed to verify paired audit evidence."
}
Write-Host "Private paired acoustic/text review comparison: $comparePath"
if (@($records | Where-Object { -not $_.independent_der_path }).Count -gt 0) {
    throw "Some Phase 1A ablations did not produce acoustic evidence."
}
Write-Host "Five Phase 1A candidate replays produced diagnostic evidence; review actual CER/DER before selecting."

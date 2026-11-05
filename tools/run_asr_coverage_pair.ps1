# Run truly paired TS3003b content-first and quality-candidate experiments.
# Quality failures are data, not successful acceptance. Nothing auto-promotes.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [string]$HumanReferenceText = "",
    [switch]$HumanReferenceVerified,
    [string]$PairId = ""
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run this private GPU experiment on the E-drive test worktree."
}
if (-not [IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
$root = Join-Path $repo "benchmark\fixed195\ami_shadow10"
if (-not $HumanReferenceText) {
    $HumanReferenceText = Join-Path $root "TS3003b_1500_1740_official_orthographic.txt"
}
$HumanReferenceText = (Resolve-Path -LiteralPath $HumanReferenceText).Path
$private = (Resolve-Path (Join-Path $repo "benchmark\fixed195")).Path.TrimEnd('\') + '\'
if (-not $HumanReferenceText.StartsWith($private, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The text reference must remain in private Fixed195 data."
}
if ($PairId -and $PairId -notmatch '^[A-Za-z0-9_-]{1,42}$') {
    throw "PairId must be a safe local experiment label."
}
$label = if ($PairId) { $PairId } else {
    (Get-Date -Format "yyyyMMdd-HHmmssfff") + "-$PID"
}
$sha = (Get-FileHash -LiteralPath $HumanReferenceText -Algorithm SHA256).Hash.ToLowerInvariant()
$cases = @(
    @{ Label = "baseline"; Mode = "ContentFirst" },
    @{ Label = "quality"; Mode = "AsrQualityCandidate" }
)
foreach ($case in $cases) {
    $tag = "$label-$($case.Label)"
    $paths = @(
        (Join-Path $root "replay_TS3003b_nemotron_$tag.json"),
        (Join-Path $root "audit_TS3003b_nemotron_$tag.json"),
        (Join-Path $root "audit_TS3003b_asr_stages_$tag.json"),
        (Join-Path $root "audit_TS3003b_phase1_human_text_$tag.json")
    )
    if (@($paths | Where-Object { Test-Path -LiteralPath $_ }).Count -ne 0) {
        throw "Refusing to overwrite a previous paired run: $tag"
    }
}
foreach ($case in $cases) {
    $tag = "$label-$($case.Label)"
    Write-Host "Paired ASR experiment: $tag"
    $arguments = @{
        PythonExe = $PythonExe
        Port = $Port
        SkipUnitTests = $true
        ExportPrivateTranscript = $true
        HumanReferenceText = $HumanReferenceText
        ExpectedReferenceSha256 = $sha
        OutputTag = $tag
    }
    $arguments[$case.Mode] = $true
    if ($HumanReferenceVerified.IsPresent) {
        $arguments.HumanReferenceVerified = $true
    }
    try {
        & (Join-Path $PSScriptRoot "run_nemotron_ami.ps1") @arguments
    } catch {
        Write-Warning ("Quality thresholds not accepted, or replay failed: " + $_.Exception.Message)
    }
    foreach ($kind in @(
        "audit_TS3003b_nemotron_",
        "audit_TS3003b_asr_stages_",
        "audit_TS3003b_phase1_human_text_"
    )) {
        $expected = Join-Path $root ("$kind$tag.json")
        if (-not (Test-Path -LiteralPath $expected -PathType Leaf)) {
            throw "Incomplete $tag experimental evidence: $expected"
        }
    }
}
$report = Join-Path $root ("phase1a_quality_pair_$label.json")
& $PythonExe -m tools.phase1a_quality_pair `
    --root $root --baseline-tag "$label-baseline" `
    --candidate-tag "$label-quality" --output $report
if ($LASTEXITCODE -ne 0) {
    throw "Cannot safely compare the frozen experiments."
}
Write-Host "Private paired baseline/quality comparison: $report"
Write-Host "Metrics are screening evidence only. Do not change default model settings."

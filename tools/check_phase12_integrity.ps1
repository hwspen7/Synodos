# Deterministic Phase 1/2 code-integrity check without downloads or JSON spam.
# This does not claim real-audio DER or cross-meeting identity acceptance.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not [IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $repo $PythonExe
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python interpreter missing: $PythonExe"
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
$private = Join-Path $repo "benchmark\fixed195"
if (-not (Test-Path -LiteralPath $private -PathType Container)) {
    New-Item -ItemType Directory -Path $private -Force | Out-Null
}
$id = (Get-Date -Format "yyyyMMdd-HHmmssfff") + "-$PID"
$log = Join-Path $private "phase12_integrity_$id.log"
$modules = @(
    "tests.quality.test_speaker_stage1_compare",
    "tests.quality.test_phase1b_diarization",
    "tests.quality.test_speaker_turn_integrity",
    "tests.quality.test_speaker_clustering_receipt",
    "tests.quality.test_ami_nemotron_live_score",
    "tests.backend.unit.test_phase2_identity_matcher",
    "tests.backend.unit.test_phase2_identity_consensus",
    "tests.backend.unit.test_phase2_speaker_store_integrity",
    "tests.backend.integration.test_phase2_manual_review",
    "tests.backend.integration.test_phase2_verified_enrollment",
    "tests.quality.test_phase2_build_trials",
    "tests.quality.test_phase2_cross_session_acceptance"
)
Write-Host "Meeting Phase 1+2 Code Integrity"
Write-Host ("Checking {0} independent test modules..." -f $modules.Count)
& $PythonExe -m unittest @modules -q *> $log
$testCode = $LASTEXITCODE
if ($testCode -ne 0) {
    Write-Host "Result: FAIL"
    Write-Host "Last diagnostic lines:"
    Get-Content -LiteralPath $log -Tail 35
    throw "Phase 1/2 deterministic tests failed. Full private log: $log"
}
Write-Host "Result: PASS (deterministic source/contract checks)"
Write-Host "Real GPU comparison: NOT RUN by this command"
Write-Host "Ten-meeting reference-backed DER: NOT CERTIFIED"
Write-Host "Cross-meeting human-verified FAR/FRR: NOT CERTIFIED"
Write-Host ("Full private log: " + $log)

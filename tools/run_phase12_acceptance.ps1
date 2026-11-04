# Run the Phase 1 and Phase 2 release preflight on the Windows test PC.
# All evidence stays under the private E-drive workspace.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [int]$Port = 8012,
    [switch]$HumanReferenceVerified,
    [string]$Fixed195Results = ""
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

Write-Host "Phase 2: deterministic identity and manual-review tests"
$modules = @(
    "tests.backend.unit.test_phase2_identity_matcher",
    "tests.backend.integration.test_phase2_manual_review",
    "tests.multilingual.test_corrections",
    "tests.quality.test_ami_human_text_reference"
)
& $PythonExe -m unittest @modules -v
if ($LASTEXITCODE -ne 0) {
    throw "Phase 2 regression tests FAILED. Phase 1 replay was not started."
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

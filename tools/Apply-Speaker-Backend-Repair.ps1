param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "Repairing meeting speaker backend in $ProjectRoot"
& $PythonExe (Join-Path $PSScriptRoot "repair_speaker_scope.py") --root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Backend scope repair failed."
}

Push-Location $ProjectRoot
try {
    & $PythonExe -m py_compile "sessions/processor.py" "sessions/session.py" "sessions/stable_speaker_channels.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Backend Python syntax check failed."
    }
    & $PythonExe -m unittest tests.backend.test_stable_speaker_channels tests.backend.test_speaker_display_projection -v
    if ($LASTEXITCODE -ne 0) {
        throw "Speaker channel regression tests failed."
    }
} finally {
    Pop-Location
}

Write-Host "Backend scope repair and deterministic speaker tests completed."

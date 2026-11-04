# Offline Windows PowerShell 5.1 path regression for GitHub CI evidence.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\..\tools\resolve_private_ci_path.ps1")

$workspace = Join-Path $env:TEMP ("meeting-ci-report-path-" + [guid]::NewGuid().ToString("N"))
$root = Join-Path $workspace "benchmark\fixed195"
New-Item -ItemType Directory -Path $root -Force | Out-Null

function Assert-Fails([scriptblock]$Code) {
    $failed = $false
    try {
        & $Code | Out-Null
    } catch {
        $failed = $true
    }
    if (-not $failed) { throw "Unsafe report path was accepted" }
}

try {
    $absolute = Join-Path $root "ci_diagnostic_absolute.json"
    $relative = "benchmark\fixed195\ci_diagnostic_relative.json"
    $expectedRelative = Join-Path $root "ci_diagnostic_relative.json"

    $actualAbsolute = Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath $absolute
    if ($actualAbsolute -ne $absolute) {
        throw "An already rooted Windows path was corrupted"
    }
    $actualRelative = Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath $relative
    if ($actualRelative -ne $expectedRelative) {
        throw "A relative Windows path was not normalized correctly"
    }
    Assert-Fails { Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath "..\outside.json" }
    Assert-Fails { Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath (Join-Path $workspace "outside.json") }
    Assert-Fails { Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath (Join-Path $root "diagnostic.txt") }

    [System.IO.File]::WriteAllText($absolute, "{}")
    Assert-Fails { Resolve-PrivateCiReportPath -Workspace $workspace -RequestedPath $absolute }
    Write-Host "CI diagnostic Windows path tests PASSED"
} finally {
    if (Test-Path $workspace) {
        Remove-Item -LiteralPath $workspace -Recurse -Force
    }
}

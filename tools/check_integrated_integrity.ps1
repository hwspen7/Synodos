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
    "tests.quality.speakers.test_speaker_stage1_compare",
    "tests.quality.recognition.test_diarization",
    "tests.quality.speakers.test_speaker_turn_integrity",
    "tests.quality.speakers.test_speaker_clustering_receipt",
    "tests.quality.ami.test_ami_nemotron_live_score",
    "tests.backend.unit.test_identity_matcher",
    "tests.backend.unit.test_identity_consensus",
    "tests.backend.unit.test_speaker_store_integrity",
    "tests.backend.integration.test_manual_review",
    "tests.backend.integration.test_verified_enrollment",
    "tests.quality.integrity.test_identity_trial_builder",
    "tests.quality.integrity.test_cross_session_acceptance"
)
Write-Host "Meeting Phase 1+2 Code Integrity"
Write-Host ("Checking {0} independent test modules..." -f $modules.Count)
# On Windows PowerShell 5.1, native stderr redirected via *> can be
# promoted to a NativeCommandError when ErrorActionPreference is Stop.
# unittest intentionally logs rejected speaker partitions to stderr even
# when every assertion passes. Only the child process exit code is authoritative.
$stdoutLog = Join-Path $private "phase12_integrity_$id.stdout.tmp"
$stderrLog = Join-Path $private "phase12_integrity_$id.stderr.tmp"
$testCode = $null
try {
    $arguments = @("-m", "unittest") + $modules + @("-q")
    $process = Start-Process -FilePath $PythonExe -ArgumentList $arguments `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    $testCode = $process.ExitCode
} finally {
    # Keep complete diagnostic data on the private benchmark drive, but
    # never print full unittest output or structured model evidence.
    $writer = [System.IO.StreamWriter]::new($log, $false, [System.Text.UTF8Encoding]::new($false))
    try {
        foreach ($entry in @(
            @{ Path = $stderrLog; Label = "Python stderr (unittest diagnostics)" },
            @{ Path = $stdoutLog; Label = "Python stdout" }
        )) {
            $writer.WriteLine("=== " + $entry.Label + " ===")
            if (Test-Path -LiteralPath $entry.Path -PathType Leaf) {
                $writer.Write((Get-Content -LiteralPath $entry.Path -Raw))
                $writer.WriteLine()
            }
        }
    } finally {
        $writer.Dispose()
        Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
    }
}
if ($null -eq $testCode -or $testCode -ne 0) {
    Write-Host "Result: FAIL (Python exit code: $testCode)"
    Write-Host "Last diagnostic lines:"
    Get-Content -LiteralPath $log -Tail 35
    throw "Phase 1/2 deterministic tests failed. Full private log: $log"
}
Write-Host "Result: PASS (deterministic source/contract checks)"
Write-Host "Real GPU comparison: NOT RUN by this command"
Write-Host "Ten-meeting reference-backed DER: NOT CERTIFIED"
Write-Host "Cross-meeting human-verified FAR/FRR: NOT CERTIFIED"
Write-Host ("Full private log: " + $log)

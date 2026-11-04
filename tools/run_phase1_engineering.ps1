# Local Phase 1 engineering acceptance for Python, Vue, and transcript safety.
# E-drive only. Never resets git, removes benchmark data, launches GPU,
# modifies a database, or merges into main.

[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$NoFrontendInstall,
    [string]$ReferenceReplay = "",
    [string]$Fixed195Results = "",
    [string]$ExpectedAudioSha256 = "",
    [string]$ExpectedReferenceSha256 = "",
    [switch]$HumanReferenceVerified
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $root.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Phase 1 acceptance requires the E-drive worktree."
}
if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $root $PythonExe
}
if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}
$PythonExe = (Resolve-Path $PythonExe).Path

$cache = Join-Path $root "benchmark\fixed195\.runtime_cache"
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$cacheNames = @{
    "HF_HOME" = "huggingface"
    "HF_HUB_CACHE" = "huggingface\hub"
    "HF_XET_CACHE" = "huggingface\xet"
    "TORCH_HOME" = "torch"
    "XDG_CACHE_HOME" = "xdg"
    "TEMP" = "temp"
    "TMP" = "temp"
    "TMPDIR" = "temp"
    "PIP_CACHE_DIR" = "pip"
    "COREPACK_HOME" = "corepack"
    "NPM_CONFIG_CACHE" = "npm"
    "PNPM_HOME" = "pnpm-home"
}
foreach ($entry in $cacheNames.GetEnumerator()) {
    $path = Join-Path $cache $entry.Value
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $path, "Process")
}
$pnpmStore = Join-Path $cache "pnpm-store"
New-Item -ItemType Directory -Path $pnpmStore -Force | Out-Null

function Require-Pass([string]$Description) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
    Write-Host "$Description PASS"
}

Write-Host "Phase 1 contract checks from $root"
& $PythonExe -c "import sys; assert sys.version_info >= (3, 12); print(sys.executable)"
Require-Pass "Python 3.12 check"

& $PythonExe -m compileall -q main.py api audio core foundation inference intelligence observability phase07 sessions speech storage tools example ops tests
Require-Pass "Python compilation"

$phase1Tests = @(
    "tests.quality.test_phase1_text_acceptance",
    "tests.quality.test_phase1_fixed195_text_gate",
    "tests.quality.test_content_first_projection",
    "tests.quality.test_ami_content_first_projection",
    "tests.quality.test_overlap_content_risk",
    "tests.quality.test_ami_offline_display_overlap",
    "tests.quality.test_live_caption_segment_audit"
)
& $PythonExe -m unittest @phase1Tests -v
Require-Pass "Phase 1 content and persistence regression"

& $PythonExe -m unittest discover -s tests -t . -v
Require-Pass "Full Python regression"

$null = Get-Command "node.exe" -ErrorAction Stop
$pnpmCmd = Get-Command "pnpm.cmd" -ErrorAction SilentlyContinue
$corepackCmd = Get-Command "corepack.cmd" -ErrorAction SilentlyContinue
if ($null -ne $pnpmCmd) {
    $launcher = $pnpmCmd.Source
    $prefix = @()
} elseif ($null -ne $corepackCmd) {
    $launcher = $corepackCmd.Source
    $prefix = @("pnpm")
} else {
    throw "pnpm.cmd or corepack.cmd is required for frontend acceptance."
}

if (-not $NoFrontendInstall.IsPresent) {
    & $launcher @prefix install --frozen-lockfile --store-dir $pnpmStore
    Require-Pass "Frozen pnpm dependency installation"
} elseif (-not (Test-Path ".\node_modules")) {
    throw "node_modules absent: omit -NoFrontendInstall to install on E."
}

& $launcher @prefix type-check
Require-Pass "Vue/TypeScript type-check"
& $launcher @prefix build-only
Require-Pass "Frontend production build"
$frontTests = @(Get-ChildItem -Path "tests/frontend/contracts" -Filter "*.test.mjs" -File | ForEach-Object { $_.FullName })
if ($frontTests.Count -lt 1) {
    throw "No frontend contract test files found"
}
& node.exe --test @frontTests
Require-Pass "Frontend transcript/WebSocket regression"

$toParse = @(
    "tools/run_nemotron_ami.ps1",
    "tools/run_phase1_engineering.ps1",
    "tools/windows/Start-Meeting.ps1"
)
foreach ($file in $toParse) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        (Resolve-Path $file).Path, [ref]$tokens, [ref]$errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        $errors | Format-List -Force
        throw "PowerShell parse errors in $file"
    }
}
Write-Host "PowerShell parser regression PASS"

if ($Fixed195Results) {
    $fixedTag = Get-Date -Format "yyyyMMdd-HHmmss"
    $fixedOutput = Join-Path $root "benchmark\fixed195\phase1_130_text_gate_$fixedTag.json"
    & $PythonExe -m tools.phase1_fixed195_text_gate --root "benchmark/fixed195" --results $Fixed195Results --output $fixedOutput
    Require-Pass "Fixed195 human-referenced 130-clip content gate"
    Write-Host "Fixed195 content report: $fixedOutput"
} else {
    Write-Host "Fixed195 130-reference text gate NOT RUN (no results supplied)."
}

if ($ReferenceReplay) {
    if (-not $ExpectedAudioSha256 -or -not $ExpectedReferenceSha256) {
        throw "Referenced text gate requires frozen audio/reference SHA256 pins."
    }
    $tag = Get-Date -Format "yyyyMMdd-HHmmss"
    $report = Join-Path $root "benchmark\fixed195\phase1_text_gate_$tag.json"
    $arguments = @(
        "-m", "tools.phase1_text_acceptance",
        "--replay", $ReferenceReplay,
        "--audio-sha256", $ExpectedAudioSha256,
        "--reference-sha256", $ExpectedReferenceSha256,
        "--output", $report
    )
    if ($HumanReferenceVerified.IsPresent) {
        $arguments += "--human-reference-verified"
    }
    & $PythonExe @arguments
    Require-Pass "Human-reference text integrity gate"
    Write-Host "Human text gate report: $report"
} else {
    Write-Host "Human-reference CER/WER gate NOT RUN (no reference-scored replay supplied)."
    Write-Host "Engineering code checks do not certify transcript content accuracy."
}

Write-Host "Phase 1 local engineering checks PASS; GitHub CI and real text accuracy are independently verified."

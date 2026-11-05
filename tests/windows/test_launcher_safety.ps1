# Validate process identity guards without terminating any real processes.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$launcher = Get-Content -Raw -LiteralPath (Join-Path $root 'tools\windows\Start-Meeting.ps1')
$main = [regex]::Match($launcher, '(?m)^try \{\r?\n    Set-Location -LiteralPath \$RepoRoot')
if (-not $main.Success) { throw 'Launcher entrypoint not found.' }
$definitions = $launcher.Substring(0, $main.Index)
$definitions = $definitions.Replace(
    '$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot ''..\..'')).Path',
    '$RepoRoot = ''' + $root + ''''
)
. ([scriptblock]::Create($definitions))

$script:OwnerName = 'python.exe'
$script:OwnerCommand = 'python -m uvicorn main:app --host 127.0.0.1 --port 8000'
$script:ValidBackend = $true
$script:ValidFrontend = $true

function Get-ProcessInfo([int]$ProcessId) {
    return [pscustomobject]@{ Name = $script:OwnerName; CommandLine = $script:OwnerCommand }
}

function Invoke-RestMethod {
    param([string]$Uri, [int]$TimeoutSec)
    if ($Uri -notlike '*/openapi.json') { throw 'Unexpected HTTP request.' }
    $title = if ($script:ValidBackend) { 'Meeting of Pioneers' } else { 'Another Application' }
    $paths = [pscustomobject]@{
        '/api/health' = [pscustomobject]@{}
        '/api/intelligence/capabilities' = [pscustomobject]@{}
    }
    return [pscustomobject]@{ info = [pscustomobject]@{ title = $title }; paths = $paths }
}

function Invoke-WebRequest {
    param([string]$Uri, [int]$TimeoutSec, [switch]$UseBasicParsing)
    $name = if ($script:ValidFrontend) { 'Meeting of Pioneers' } else { 'Another Application' }
    return [pscustomobject]@{
        StatusCode = 200
        Content = '<html><title>' + $name + '</title><script src="/src/main.ts"></script></html>'
    }
}

function Expect-Denied([scriptblock]$Action, [string]$Description) {
    $denied = $false
    try { & $Action } catch { $denied = $true }
    if (-not $denied) { throw "Unsafe process accepted: $Description" }
    Write-Output "Rejected as expected: $Description"
}

# A relative working-directory invocation does not expose the repository
# absolute path in its command line, but Meeting's API identifies it.
Assert-ProjectProcess 13776 'uvicorn' 8000
Assert-ProjectProcess 13776 'main:app' 8000

$script:ValidBackend = $false
Expect-Denied { Assert-ProjectProcess 13776 'uvicorn' 8000 } 'unrelated HTTP service'
$script:ValidBackend = $true
$script:OwnerName = 'notepad.exe'
Expect-Denied { Assert-ProjectProcess 13776 'uvicorn' 8000 } 'non-Python process'

$script:OwnerName = 'python.exe'
$script:OwnerCommand = "$RepoRoot\.venv\Scripts\python.exe -m uvicorn main:app"
$script:ValidBackend = $false
Assert-ProjectProcess 13776 'uvicorn' 8000

$script:OwnerName = 'node.exe'
$script:OwnerCommand = 'node vite.js'
$script:ValidFrontend = $true
Assert-ProjectProcess 9876 'vite' 5173
$script:ValidFrontend = $false
Expect-Denied { Assert-ProjectProcess 9876 'vite' 5173 } 'unrelated frontend'

# Guard against regressions where a child frontend window unconditionally
# invokes Vite despite a previously bound Meeting frontend port.
if ($launcher -notmatch [regex]::Escape("$" + "listeningPid = Get-ListeningPid 5173")) {
    throw 'Frontend child process lacks its own port check.'
}
if ($launcher -notmatch 'Local\\MeetingOfPioneers_Vite_5173') {
    throw 'Frontend child process lacks the cross-process single-start gate.'
}
if ($launcher -notmatch 'Another Meeting frontend is initializing') {
    throw 'Frontend child process lacks a bounded competing-launch wait.'
}
# The multi-speaker profile must be a deliberate opt-in and must not reuse a
# pre-existing single-speaker backend with the wrong process environment.
if ($launcher -notmatch 'MEETING_MULTISPEAKER_TEST' -or
    $launcher -notmatch "MEETING_INDEPENDENT_DIARIZATION = 'nemotron3'" -or
    $launcher -notmatch "MEETING_DIARIZATION_MODE = 'cluster'") {
    throw 'Explicit multi-speaker model activation is missing.'
}
if ($launcher -notmatch "MEETING_FORCE_BACKEND_RESTART -eq '1' -or") {
    throw 'Multi-speaker mode must restart any previous backend.'
}
Write-Output 'Windows launcher process safety tests passed.'

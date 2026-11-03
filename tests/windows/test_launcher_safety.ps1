# Validate process identity guards without terminating any real processes.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$launcher = Get-Content -Raw -LiteralPath (Join-Path $root 'tools\windows\Start-Meeting.ps1')
$main = [regex]::Match($launcher, '(?m)^try \{\r?\n    Set-Location -LiteralPath \$RepoRoot')
if (-not $main.Success) { throw 'Launcher entrypoint not found.' }
. ([scriptblock]::Create($launcher.Substring(0, $main.Index)))

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

Write-Output 'Windows launcher process safety tests passed.'

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Clients = 2,
    [int]$Sessions = 4,
    [int]$Seconds = 12
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Python virtual environment not found.' }
& $python -m ops.soak_test --clients $Clients --sessions $Sessions --seconds $Seconds --confirm-test-database
if ($LASTEXITCODE -ne 0) { throw 'Live soak test failed; inspect backend logs and separate test database.' }

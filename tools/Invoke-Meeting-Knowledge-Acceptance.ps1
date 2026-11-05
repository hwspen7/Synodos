# Run both actual local Ollama Meeting knowledge acceptance modes.
# Uses existing environment credentials; never prints DATABASE_URL or password.
param(
    [string]$Python = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe",
    [string]$ReportRoot = "E:\AI-Cache\MeetingResults",
    [switch]$LexicalOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}
if (-not $env:DATABASE_URL) {
    throw "DATABASE_URL is missing in this PowerShell session."
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:MEETING_STORAGE_BACKEND = "postgres"
$env:MEETING_AI_ENABLED = "1"
$env:MEETING_AI_PROVIDER = "ollama"
$env:MEETING_AI_ENDPOINT = "http://127.0.0.1:11434"
$env:MEETING_AI_MEDIA_ROOT = "E:\AI-Cache\MeetingMedia"
$env:PIP_CACHE_DIR = "E:\AI-Cache\pip"
$env:TEMP = "E:\AI-Cache\tmp"
$env:TMP = "E:\AI-Cache\tmp"

New-Item -ItemType Directory -Force $ReportRoot, $env:MEETING_AI_MEDIA_ROOT, $env:TEMP | Out-Null

$Tags = Invoke-RestMethod "$env:MEETING_AI_ENDPOINT/api/tags"
$Installed = @($Tags.models | ForEach-Object { $_.name })

if (-not $env:MEETING_AI_TEXT_MODEL) {
    $Saluki = @($Installed | Where-Object { $_ -match 'Saluki-27B' })
    if ($Saluki.Count -eq 0) {
        throw "Saluki 27B is not listed in local Ollama. Configure MEETING_AI_TEXT_MODEL."
    }
    $env:MEETING_AI_TEXT_MODEL = $Saluki[0]
}
if ($env:MEETING_AI_TEXT_MODEL -notin $Installed) {
    throw "Configured text model is not installed in Ollama."
}
if (-not $env:MEETING_AI_EMBED_MODEL) {
    $env:MEETING_AI_EMBED_MODEL = "qwen3-embedding:0.6b"
}

Write-Host "Meeting knowledge local acceptance (lexical mode)"
Write-Host "Text model: $env:MEETING_AI_TEXT_MODEL"
$env:MEETING_AI_EMBEDDINGS = "0"
$LexicalReport = Join-Path $ReportRoot "meeting_knowledge_lexical.json"
& $Python -m tools.meeting_knowledge_synthetic --mode lexical --output $LexicalReport
$LexicalExit = $LASTEXITCODE

$HybridExit = 0
if (-not $LexicalOnly) {
    if ($env:MEETING_AI_EMBED_MODEL -notin $Installed) {
        Write-Warning "Embedding model missing: $env:MEETING_AI_EMBED_MODEL"
        Write-Warning "No model will be downloaded automatically. Install the embedding model to your E-drive Ollama cache."
        $HybridExit = 2
    }
    else {
        Write-Host "Meeting knowledge local acceptance (hybrid mode)"
        $env:MEETING_AI_EMBEDDINGS = "1"
        $HybridReport = Join-Path $ReportRoot "meeting_knowledge_hybrid.json"
        & $Python -m tools.meeting_knowledge_synthetic --mode hybrid --output $HybridReport
        $HybridExit = $LASTEXITCODE
    }
}

Write-Host "Lexical acceptance exit code: $LexicalExit"
Write-Host "Hybrid acceptance exit code: $HybridExit"
Write-Host "Reports: $ReportRoot"

if ($LexicalExit -ne 0 -or $HybridExit -ne 0) {
    throw "Meeting knowledge local acceptance was not fully passed. Inspect the saved reports."
}
Write-Host "MEETING KNOWLEDGE LOCAL SYNTHETIC ACCEPTANCE: PASS"

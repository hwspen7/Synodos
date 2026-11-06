# Run both actual local Ollama Meeting knowledge acceptance modes.
# Uses existing environment credentials; never prints DATABASE_URL or password.
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$ReportRoot = "E:\AI-Cache\MeetingResults",
    [string]$Endpoint = "",
    [switch]$LexicalOnly,
    [switch]$HybridOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($LexicalOnly -and $HybridOnly) {
    throw "LexicalOnly and HybridOnly are mutually exclusive."
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
if (-not $Python) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if (-not $env:DATABASE_URL) {
    throw "DATABASE_URL is missing in this PowerShell session."
}
if (-not $Endpoint) {
    $Endpoint = if ($env:MEETING_AI_ENDPOINT) {
        $env:MEETING_AI_ENDPOINT
    } else {
        "http://127.0.0.1:11434"
    }
}
try { $Uri = [uri]$Endpoint } catch { throw "Invalid local Ollama endpoint: $Endpoint" }
if ($Uri.Scheme -ne "http" -or $Uri.Host -notin @("127.0.0.1","localhost","::1") -or
    $Uri.AbsolutePath -ne "/" -or $Uri.Query -or $Uri.UserInfo) {
    throw "Only a local Ollama HTTP endpoint without credentials or path is supported."
}
$Endpoint = $Endpoint.TrimEnd("/")

$env:MEETING_STORAGE_BACKEND = "postgres"
$env:MEETING_AI_ENABLED = "1"
$env:MEETING_AI_PROVIDER = "ollama"
$env:MEETING_AI_ENDPOINT = $Endpoint
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

Write-Host "Ollama endpoint: $Endpoint"
Write-Host "Text model: $env:MEETING_AI_TEXT_MODEL"
Write-Host "Embedding model: $env:MEETING_AI_EMBED_MODEL"

# Check the model BEFORE repeating expensive, already-passed lexical probes.
if (-not $LexicalOnly -and $env:MEETING_AI_EMBED_MODEL -notin $Installed) {
    Write-Warning "Embedding model missing at this Ollama endpoint: $env:MEETING_AI_EMBED_MODEL"
    Write-Warning "The existing lexical PASS remains valid. No hybrid test was run."
    Write-Warning "Install the model into the running Ollama server's E-drive model store."
    Write-Warning "For an Ollama CLI connected to this server, set OLLAMA_HOST to $($Uri.Authority) and run: ollama pull $env:MEETING_AI_EMBED_MODEL"
    Write-Warning "OLLAMA_MODELS must be set in the SERVER environment before starting Ollama, not only in this client shell."
    throw "HYBRID NOT_ACCEPTED: embedding model is unavailable. No report was fabricated."
}

$LexicalReport = Join-Path $ReportRoot "meeting_knowledge_lexical.json"
$HybridReport = Join-Path $ReportRoot "meeting_knowledge_hybrid.json"
if (-not $HybridOnly) {
    Write-Host "Meeting knowledge local acceptance (lexical mode)"
    $env:MEETING_AI_EMBEDDINGS = "0"
    & $Python -m tools.maintenance.meeting_knowledge_synthetic --mode lexical --output $LexicalReport
    if ($LASTEXITCODE -ne 0) {
        throw "Lexical acceptance failed. Inspect $LexicalReport"
    }
}
if ($LexicalOnly) {
    Write-Host "LEXICAL ACCEPTANCE: PASS. HYBRID NOT_RUN. Full synthetic acceptance is NOT_ACCEPTED."
    exit 0
}
Write-Host "Meeting knowledge local acceptance (hybrid mode)"
$env:MEETING_AI_EMBEDDINGS = "1"
& $Python -m tools.maintenance.meeting_knowledge_synthetic --mode hybrid --output $HybridReport
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid acceptance failed. Inspect $HybridReport"
}

# Verify both complete reports, including individual cases and exact model IDs.
# HybridOnly must never certify a missing or stale lexical report.
if (-not (Test-Path -LiteralPath $LexicalReport)) {
    throw "Lexical report missing: $LexicalReport. Run -LexicalOnly first."
}
& $Python -m tools.validation.meeting_release_gate verify --lexical $LexicalReport --hybrid $HybridReport
if ($LASTEXITCODE -ne 0) {
    throw "Combined lexical/hybrid verification failed. See saved reports."
}
Write-Host "MEETING KNOWLEDGE LOCAL SYNTHETIC ACCEPTANCE: PASS"
Write-Host "Reports: $ReportRoot"

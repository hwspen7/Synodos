# One-command Meeting knowledge baseline freeze, real-audio gate, and release decision.
# No installs, model substitutions, database deletions or automatic tag/merge.
[CmdletBinding()]
param(
    [string]$Python = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe",
    [string]$ReportRoot = "E:\AI-Cache\MeetingResults",
    [string]$GoldManifest = "E:\AI-Cache\MeetingResults\meeting_audio_review_gold.json",
    [string]$AudioIdentityGate = "",
    [string]$HostedCI = "",
    [switch]$ReindexTestMeetings
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not (Test-Path -LiteralPath $Python)) { throw "Python interpreter missing" }
if (-not $env:DATABASE_URL) { throw "DATABASE_URL must point to meeting_test" }
if (-not $repo.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Run from E-drive test worktree only"
}
$env:MEETING_STORAGE_BACKEND = "postgres"
$env:MEETING_AI_ENABLED = "1"
$env:MEETING_AI_PROVIDER = "ollama"
$env:MEETING_AI_ENDPOINT = "http://127.0.0.1:11434"
$env:MEETING_AI_EMBED_MODEL = if ($env:MEETING_AI_EMBED_MODEL) { $env:MEETING_AI_EMBED_MODEL } else { "qwen3-embedding:0.6b" }
$env:MEETING_AI_MEDIA_ROOT = "E:\AI-Cache\MeetingMedia"
$env:MEETING_SOURCE_COMMIT = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $env:MEETING_SOURCE_COMMIT -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve exact Git commit"
}
New-Item -Path $ReportRoot -ItemType Directory -Force | Out-Null
$tag = Get-Date -Format "yyyyMMdd-HHmmss-ffff"
$baseline = Join-Path $ReportRoot "meeting_knowledge_baseline_$tag.json"
$lex = Join-Path $ReportRoot "meeting_knowledge_lexical.json"
$hyb = Join-Path $ReportRoot "meeting_knowledge_hybrid.json"
# Accept previously generated successful reports without overwriting them.
if (-not (Test-Path -LiteralPath $lex)) {
    $lex = Join-Path $ReportRoot 'phase34_lexical.json'
}
if (-not (Test-Path -LiteralPath $hyb)) {
    $hyb = Join-Path $ReportRoot 'phase34_hybrid.json'
}
& $Python -m tools.meeting_release_gate freeze --lexical $lex --hybrid $hyb --output $baseline
if ($LASTEXITCODE -ne 0) { throw "Synthetic baseline failed strict freeze" }
Write-Host "Frozen synthetic baseline: $baseline"
$frozen = Get-Content -LiteralPath $baseline -Raw | ConvertFrom-Json
if (-not $env:MEETING_AI_TEXT_MODEL) { $env:MEETING_AI_TEXT_MODEL = $frozen.text_model }
if ($env:MEETING_AI_TEXT_MODEL -ne $frozen.text_model) {
    throw "Configured Ollama text model differs from frozen synthetic baseline"
}
if ($env:MEETING_AI_EMBED_MODEL -ne $frozen.embedding_model) {
    throw "Configured embedding model differs from frozen synthetic baseline"
}
$installed = @((Invoke-RestMethod "$env:MEETING_AI_ENDPOINT/api/tags").models | ForEach-Object { $_.name })
if ($env:MEETING_AI_TEXT_MODEL -notin $installed -or $env:MEETING_AI_EMBED_MODEL -notin $installed) {
    throw "The frozen text or embedding model is not installed in local Ollama"
}
$realLex = Join-Path $ReportRoot "meeting_real_lexical_$tag.json"
$realHyb = Join-Path $ReportRoot "meeting_real_hybrid_$tag.json"
if (Test-Path -LiteralPath $GoldManifest) {
    $extra = @()
    if ($ReindexTestMeetings) { $extra += "--reindex" }
    $env:MEETING_AI_EMBEDDINGS = "0"
    & $Python -m tools.meeting_knowledge_benchmark --manifest $GoldManifest --output $realLex --mode lexical @extra
    $lexExit = $LASTEXITCODE
    $env:MEETING_AI_EMBEDDINGS = "1"
    & $Python -m tools.meeting_knowledge_benchmark --manifest $GoldManifest --output $realHyb --mode hybrid @extra
    $hybExit = $LASTEXITCODE
    if ($lexExit -ne 0 -or $hybExit -ne 0) {
        Write-Warning "Real meeting quality tests did not pass. Release remains blocked."
    }
} else {
    Write-Warning "Human-reviewed real audio gold manifest missing: $GoldManifest"
    Write-Warning "No fake real-meeting PASS will be generated. See docs/meeting_knowledge_quality.md"
}
$decision = Join-Path $ReportRoot "meeting_release_decision_$tag.json"
$argsList = @("-m", "tools.meeting_release_gate", "decide", "--baseline", $baseline,
              "--real-lexical", $realLex, "--real-hybrid", $realHyb,
              "--commit", $env:MEETING_SOURCE_COMMIT, "--output", $decision)
if ($AudioIdentityGate) { $argsList += @("--audio-identity", $AudioIdentityGate) }
if ($HostedCI) { $argsList += @("--ci", $HostedCI) }
& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "Release NOT_ACCEPTED. Decision report: $decision" }
Write-Host "Release gates passed. Human final review is still required."

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Message = 'chore: checkpoint stable PostgreSQL foundation and operations tooling',
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = 'Stop'
if ($Push -and -not $Commit) { throw '-Push also requires -Commit.' }
if (-not (Test-Path (Join-Path $ProjectRoot '.git'))) { throw 'Not a Git repository root.' }
Push-Location $ProjectRoot
try {
    $branch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect Git branch.' }
    if (-not $branch -or $branch -in @('main', 'master')) {
        throw 'Refusing to commit directly on the default branch.'
    }
    $staged = & git diff --cached --name-only
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect staged changes.' }
    if (-not [string]::IsNullOrWhiteSpace(($staged -join ""))) {
        throw 'Git index already has staged changes. Commit or unstage them manually first.'
    }
    Write-Host "Active branch: $branch"
    Write-Host 'Pending changes:'
    & git status --short
    if (-not $Commit) {
        Write-Host 'PREVIEW ONLY. No files staged, committed, or pushed.'
        Write-Host 'Review the status, then rerun with -Commit to create a local checkpoint.'
        return
    }
    $allowlist = @('.github','api','audio','core','foundation','inference','observability','sessions','storage','src','tests','docs','deploy','ops','main.py','pyproject.toml','uv.lock','package.json','pnpm-lock.yaml','README.md','PROJECT_CONTEXT.md','.gitignore')
    $included = @($allowlist | Where-Object { Test-Path -LiteralPath $_ })
    if ($included.Count -eq 0) { throw 'No recognized source paths.' }
    # Do not stage arbitrary files at repository root or under model/data paths.
    & git add -A -- @included
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
    $names = @(& git diff --cached --name-only)
    $danger = @($names | Where-Object { $_ -match '(?i)(^|/)(\.env(\.|$)|[^/]*\.pkl$|[^/]*\.dump$|[^/]*\.db$|[^/]*\.sqlite$|[^/]*\.key$|[^/]*\.pem$|[^/]*pgpass[^/]*|[^/]*credential[^/]*\.(xml|json)$|[^/]*\.zip$)' })
    if ($danger.Count -gt 0) {
        & git reset --quiet HEAD -- @included
        throw 'A sensitive or generated file was detected in staging. Nothing was committed.'
    }
    # Inspect only newly added lines, and associate any DSN with its file.
    # Test URLs are assembled in fragments so this file can pass its own scan.
    $metricsFixture = 'postgresql:' + '//' + 'user:secret@localhost/db'
    $ciFixture = 'postgresql:' + '//' + 'meeting_test:ci_only_password@localhost:5432/meeting_test'
    $diff = @(& git diff --cached --unified=0 -- @included)
    if ($LASTEXITCODE -ne 0) {
        & git reset --quiet HEAD -- @included
        throw 'Unable to inspect staged source changes. Nothing was committed.'
    }
    $diffPath = ''
    foreach ($line in $diff) {
        if ($line -match '^\+\+\+ b/(.+)$') {
            $diffPath = $Matches[1].Replace('\', '/')
            continue
        }
        if (-not ($line.StartsWith('+') -and -not $line.StartsWith('+++'))) {
            continue
        }
        $addedLine = $line.Substring(1)
        $dsnMatches = [regex]::Matches($addedLine, '(?i)postgres(?:ql)?://[^\s/]+:([^\s@]+)@')
        foreach ($dsnMatch in $dsnMatches) {
            $candidate = $dsnMatch.Groups[1].Value.Trim("'", '"')
            $isPlaceholder = $candidate -match '^(?i:REPLACE_ME|YOUR_PASSWORD|PASSWORD|CHANGE_ME|REDACTED|EXAMPLE|\*+)$'
            $isTestFixture = (
                $diffPath -ceq 'tests/infrastructure/unit/test_metrics.py' -and
                $metricsFixture.StartsWith($dsnMatch.Value, [System.StringComparison]::Ordinal) -and
                $addedLine.Contains($metricsFixture)
            )
            $isCiFixture = (
                $diffPath -ceq '.github/workflows/ci.yml' -and
                $ciFixture.StartsWith($dsnMatch.Value, [System.StringComparison]::Ordinal) -and
                $addedLine.Contains($ciFixture)
            )
            if (-not ($isPlaceholder -or $isTestFixture -or $isCiFixture)) {
                & git reset --quiet HEAD -- @included
                Write-Host "Credential review required in: $diffPath" -ForegroundColor Yellow
                throw 'Possible database password found in staged diff. Nothing was committed.'
            }
        }
    }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host 'No staged source changes to commit.'; return }
    Write-Host "Committing $($names.Count) changed source files..."
    & git commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    & git log -1 --format='%h %s'
    if ($Push) {
        & git push -u origin $branch
        if ($LASTEXITCODE -ne 0) { throw 'Push failed; your local commit is preserved.' }
        Write-Host 'REMOTE PUSH COMPLETE' -ForegroundColor Green
    } else {
        Write-Host 'LOCAL CHECKPOINT COMMITTED. Remote repository unchanged.' -ForegroundColor Green
    }
} finally {
    Pop-Location
}

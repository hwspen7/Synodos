# Diagnose GitHub Actions before-job failure without masking a failing check.
# Read-only. Requires public repository access or an authenticated 'gh' CLI.
# Source, caches and diagnostic outputs remain on E: drive.

[CmdletBinding()]
param(
    [string]$Repository = "hwspen7/meeting",
    [string]$Commit = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw "Expected owner/repository"
}
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspace
if (-not $workspace.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Use the E-drive project worktree."
}
if (-not $Commit) {
    $Commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve HEAD" }
}
if ($Commit -notmatch '^[a-f0-9]{40}$') {
    throw "Commit must be a full 40-character SHA"
}

function Get-RepoApi([string]$Endpoint) {
    # GitHub's public REST API can serve check-run annotations. For a
    # private repository, GitHub CLI is used with its local authorization.
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if ($null -ne $gh) {
        $ghPath = $gh.Source
        $raw = & $ghPath api "repos/$Repository/$Endpoint" 2>&1
        if ($LASTEXITCODE -eq 0) {
            return ($raw -join "`n" | ConvertFrom-Json)
        }
    }
    $uri = "https://api.github.com/repos/$Repository/$Endpoint"
    $headers = @{
        "Accept" = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "meeting-phase1-ci-audit"
    }
    $secret = if ($env:GH_TOKEN) { $env:GH_TOKEN } else { $env:GITHUB_TOKEN }
    if ($secret) {
        $headers["Authorization"] = "Bearer $secret"
    }
    try {
        return Invoke-RestMethod -Uri $uri -Headers $headers -Method Get -TimeoutSec 20
    } catch {
        if (-not $secret -and $null -eq $gh) {
            throw "This is a private GitHub repository. Install/authenticate GitHub CLI (gh auth login), or provide GH_TOKEN in the environment. Never paste the token into logs."
        }
        throw
    }
}

$checks = Get-RepoApi "commits/$Commit/check-runs?per_page=100"
if ($null -eq $checks -or @($checks.check_runs).Count -eq 0) {
    throw "GitHub returned no check runs for this commit; cannot certify CI."
}
$rows = @()
foreach ($check in @($checks.check_runs)) {
    $annotations = @()
    if ($check.output.annotations_count -gt 0) {
        try {
            $annotations = @(
                Get-RepoApi "check-runs/$($check.id)/annotations?per_page=100"
            )
        } catch {
            $annotations = @(@{
                message = "Check-run annotations inaccessible: $($_.Exception.Message)"
            })
        }
    }
    $job = $null
    try {
        $job = Get-RepoApi "actions/jobs/$($check.id)"
    } catch {
        # Checks from third-party apps need not have a GitHub Actions job.
    }
    $messages = @($annotations | ForEach-Object {
        if ($_.message) { [string]$_.message }
    })
    $runnerId = if ($null -ne $job -and $null -ne $job.runner_id) {
        [string]$job.runner_id
    } else { "" }
    $stepCount = if ($null -ne $job) {
        if ($null -eq $job.steps) { 0 } else { @($job.steps).Count }
    } else { $null }
    $schedulerBlocked = ($check.conclusion -eq "failure" -and
        $null -ne $job -and ($runnerId -eq "" -or $runnerId -eq "0") -and
        $stepCount -eq 0)
    $billing = ($messages -join " ") -match '(billing|payment|budget|spending limit)'
    $rows += [pscustomobject]@{
        name = [string]$check.name
        conclusion = [string]$check.conclusion
        status = [string]$check.status
        github_actions_job = ($null -ne $job)
        runner_id = $runnerId
        executed_steps = $stepCount
        scheduler_blocked = [bool]$schedulerBlocked
        billing_annotation = [bool]$billing
        annotations = $messages
        html_url = [string]$check.html_url
    }
}
$counts = @($rows | Where-Object { $_.conclusion -eq "success" }).Count
$failures = @($rows | Where-Object {
    $_.conclusion -ne "success"
}).Count
$report = [ordered]@{
    schema = "meeting-phase1-github-actions-diagnostic-v1"
    repository = $Repository
    commit_sha = $Commit
    checks = $rows
    check_count = $rows.Count
    successful_checks = $counts
    failed_or_incomplete_checks = $failures
    all_required_checks_green = ($rows.Count -gt 0 -and $failures -eq 0)
    note = "Annotations diagnose check failures; this script cannot change account billing or allocate GitHub-hosted runners."
}
$report | ConvertTo-Json -Depth 12
if ($Output) {
    $private = (Join-Path $workspace "benchmark\fixed195")
    $path = [System.IO.Path]::GetFullPath(
        (Join-Path $workspace $Output)
    )
    $privateRoot = [System.IO.Path]::GetFullPath($private).TrimEnd('\') + '\'
    if (-not $path.StartsWith($privateRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        [System.IO.Path]::GetExtension($path) -ne ".json" -or
        [System.IO.File]::Exists($path)) {
        throw "Use a NEW .json under benchmark/fixed195 for CI diagnostics"
    }
    New-Item -ItemType Directory -Path (Split-Path $path -Parent) -Force | Out-Null
    [System.IO.File]::WriteAllText(
        $path, ($report | ConvertTo-Json -Depth 12),
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "Private CI diagnostic: $path"
}
if (@($rows | Where-Object { $_.billing_annotation }).Count -gt 0) {
    Write-Warning "GitHub reports a billing/budget-related job annotation. Inspect the correct OWNER account Billing & Licensing -> Budgets and alerts and payment status. If these are valid, contact GitHub Support."
}
if (@($rows | Where-Object { $_.scheduler_blocked }).Count -gt 0) {
    Write-Warning "Some jobs failed before runner allocation. There are no test-step logs; changing assertions cannot repair those check results."
}
if (-not $report.all_required_checks_green) {
    throw "GitHub check status is not green. Do NOT merge/tag Phase 1 yet."
}
Write-Host "All checks for the selected commit passed."

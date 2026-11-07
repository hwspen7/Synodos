<#
.SYNOPSIS
Configure the Git author for this Synodos checkout.
.DESCRIPTION
GitHub associates contributions with a verified email on your account.
Use the same email as the existing hwspen7 commits for future local changes.
This modifies local Git configuration only, never repository history.
#>
[CmdletBinding()]
param(
    [string]$Name = 'Spencer Qin',
    [string]$Email = 'spencerqin@outlook.com'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

& git -C $projectRoot rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'The script must run from a Git checkout.'
}

& git -C $projectRoot config --local user.name $Name
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to configure the Git author name.'
}
& git -C $projectRoot config --local user.email $Email
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to configure the Git author email.'
}

Write-Host 'Local Git author configured:'
& git -C $projectRoot config --local --get user.name
& git -C $projectRoot config --local --get user.email
Write-Host 'Existing commits are unchanged.'

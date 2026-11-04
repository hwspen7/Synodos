# Resolve a diagnostic report path without corrupting absolute Windows paths.
# Keep this helper independent of GitHub auth so it can be unit-tested offline.
function Resolve-PrivateCiReportPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Workspace,
        [Parameter(Mandatory = $true)][string]$RequestedPath
    )

    $private = [System.IO.Path]::GetFullPath(
        (Join-Path $Workspace "benchmark\fixed195")
    )
    if ([System.IO.Path]::IsPathRooted($RequestedPath)) {
        $candidate = $RequestedPath
    } else {
        $candidate = Join-Path $Workspace $RequestedPath
    }
    $path = [System.IO.Path]::GetFullPath($candidate)
    $separator = [System.IO.Path]::DirectorySeparatorChar
    $rootWithSeparator = $private.TrimEnd([char[]]@('\', '/')) + $separator
    if (-not $path.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase) -or
        [System.IO.Path]::GetExtension($path) -ne ".json" -or
        [System.IO.File]::Exists($path)) {
        throw "Use a NEW .json under benchmark/fixed195 for CI diagnostics"
    }
    return $path
}

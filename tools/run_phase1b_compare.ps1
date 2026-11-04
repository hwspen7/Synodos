# Paired three-model AMI experiment; never changes the live default model.
# Requires ten prepared frozen AMI cases and ten completed Nemotron replays.
# All source audio and model assets remain on E:.
[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$PyannotePython = "",
    [string]$DiarizenPython = "",
    [string]$NemotronReplayManifest = "",
    [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
    [string]$PyannoteModelPath = "",
    [string]$DiarizenModelPath = "",
    [string]$DiarizenEmbeddingModelPath = "",
    [switch]$AllowModelDownload,
    [string]$RunId = ""
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (-not $repo.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Phase 1B requires the E-drive benchmark worktree."
}
function Resolve-Python([string]$Value) {
    if (-not $Value) { throw "Missing dedicated Python interpreter" }
    $path = if ([IO.Path]::IsPathRooted($Value)) { $Value }
            else { Join-Path $repo $Value }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Python not found: $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}
# Keep model downloads and temporary files off the Windows system drive.
$cacheBase = Join-Path $repo "benchmark\fixed195\.runtime_cache"
foreach ($key in @("HF_HOME","HF_HUB_CACHE","HF_XET_CACHE","TORCH_HOME",
                   "MODELSCOPE_CACHE","PIP_CACHE_DIR","TEMP","TMP","TMPDIR")) {
    $old = [Environment]::GetEnvironmentVariable($key, "Process")
    if (-not $old -or -not $old.StartsWith("E:\", [StringComparison]::OrdinalIgnoreCase)) {
        $dest = Join-Path $cacheBase $key.ToLowerInvariant()
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        [Environment]::SetEnvironmentVariable($key, $dest, "Process")
    }
}
$PythonExe = Resolve-Python $PythonExe
$PyannotePython = Resolve-Python $PyannotePython
$DiarizenPython = Resolve-Python $DiarizenPython

$root = Join-Path $repo "benchmark\fixed195\ami_shadow10"
$private = (Resolve-Path (Join-Path $repo "benchmark\fixed195")).Path
if (-not $NemotronReplayManifest) {
    throw "Supply a private JSON map of meeting ID to completed Nemotron replay path."
}
$mappingFile = (Resolve-Path -LiteralPath $NemotronReplayManifest).Path
if (-not $mappingFile.StartsWith(
        ($private.TrimEnd('\') + '\'),
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Nemotron replay manifest must remain private under benchmark/fixed195."
}
$map = Get-Content -LiteralPath $mappingFile -Raw | ConvertFrom-Json
$meetings = @("EN2002a","EN2002b","EN2002d","ES2004a","ES2004c",
              "ES2004d","IS1009b","IS1009d","TS3003b","TS3003c")
foreach ($meeting in $meetings) {
    $ref = $map.PSObject.Properties[$meeting]
    if ($null -eq $ref -or -not $ref.Value) {
        throw "Missing Nemotron replay for $meeting"
    }
    $replay = [string]$ref.Value
    if (-not [IO.Path]::IsPathRooted($replay)) { $replay = Join-Path $repo $replay }
    if (-not (Test-Path -LiteralPath $replay -PathType Leaf)) {
        throw "Missing completed replay: $replay"
    }
    $case = Join-Path $root ("cases\" + $meeting + ".json")
    if (-not (Test-Path -LiteralPath $case)) {
        throw "Frozen AMI case not prepared: $case"
    }
}
if ($RunId -and $RunId -notmatch '^[a-zA-Z0-9_-]{1,64}
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
$results = @()
foreach ($meeting in $meetings) {
    foreach ($model in @("nemotron3","pyannote","diarizen")) {
        $python = switch ($model) {
            "pyannote" { $PyannotePython }
            "diarizen" { $DiarizenPython }
            default { $PythonExe }
        }
        $out = Join-Path $runDir ("$model-$meeting.json")
        $args = @("-m","tools.phase1b_diarization_benchmark",
                  "--root",$root,"--meeting",$meeting,"--model",$model,
                  "--device",$Device,"--output",$out)
        if ($model -eq "nemotron3") {
            $replay = [string]$map.PSObject.Properties[$meeting].Value
            if (-not [IO.Path]::IsPathRooted($replay)) { $replay = Join-Path $repo $replay }
            $args += @("--replay",$replay)
        } else {
            $modelPath = if ($model -eq "pyannote") { $PyannoteModelPath }
                         else { $DiarizenModelPath }
            if ($modelPath) { $args += @("--model-id",$modelPath) }
            if ($model -eq "diarizen" -and $DiarizenEmbeddingModelPath) {
                $args += @("--diarizen-embedding-model",$DiarizenEmbeddingModelPath)
            }
            if ($AllowModelDownload.IsPresent) { $args += "--allow-model-download" }
        }
        Write-Host "Phase 1B: $meeting / $model"
        & $python @args
        # Exit 2 means that a real quality metric was computed but >=20%.
        # Continue comparing all models, while preserving the failure status.
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
            throw "Model benchmark failed: $meeting / $model"
        }
        if (-not (Test-Path -LiteralPath $out)) {
            throw "Model produced no benchmark artifact: $model"
        }
        $results += $out
    }
}
$combined = Join-Path $runDir "paired_comparison.json"
& $PythonExe -m tools.phase1b_compare --reports @results --output $combined --min-meetings 10
if ($LASTEXITCODE -ne 0) {
    throw "Paired model comparison did not meet the acoustic quality gate. See $combined"
}
Write-Host "All three models paired. Review $combined; never auto-promote a model."
) {
    throw "RunId must be safe for private filenames."
}
$runId = if ($RunId) { $RunId } else { (Get-Date -Format "yyyyMMdd-HHmmss") + "-$PID" }
$runDir = Join-Path $root ("phase1b\" + $runId)
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
$results = @()
foreach ($meeting in $meetings) {
    foreach ($model in @("nemotron3","pyannote","diarizen")) {
        $python = switch ($model) {
            "pyannote" { $PyannotePython }
            "diarizen" { $DiarizenPython }
            default { $PythonExe }
        }
        $out = Join-Path $runDir ("$model-$meeting.json")
        $args = @("-m","tools.phase1b_diarization_benchmark",
                  "--root",$root,"--meeting",$meeting,"--model",$model,
                  "--device",$Device,"--output",$out)
        if ($model -eq "nemotron3") {
            $replay = [string]$map.PSObject.Properties[$meeting].Value
            if (-not [IO.Path]::IsPathRooted($replay)) { $replay = Join-Path $repo $replay }
            $args += @("--replay",$replay)
        } else {
            $modelPath = if ($model -eq "pyannote") { $PyannoteModelPath }
                         else { $DiarizenModelPath }
            if ($modelPath) { $args += @("--model-id",$modelPath) }
            if ($model -eq "diarizen" -and $DiarizenEmbeddingModelPath) {
                $args += @("--diarizen-embedding-model",$DiarizenEmbeddingModelPath)
            }
            if ($AllowModelDownload.IsPresent) { $args += "--allow-model-download" }
        }
        Write-Host "Phase 1B: $meeting / $model"
        & $python @args
        # Exit 2 means that a real quality metric was computed but >=20%.
        # Continue comparing all models, while preserving the failure status.
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
            throw "Model benchmark failed: $meeting / $model"
        }
        if (-not (Test-Path -LiteralPath $out)) {
            throw "Model produced no benchmark artifact: $model"
        }
        $results += $out
    }
}
$combined = Join-Path $runDir "paired_comparison.json"
& $PythonExe -m tools.phase1b_compare --reports @results --output $combined --min-meetings 10
if ($LASTEXITCODE -ne 0) {
    throw "Paired model comparison did not meet the acoustic quality gate. See $combined"
}
Write-Host "All three models paired. Review $combined; never auto-promote a model."

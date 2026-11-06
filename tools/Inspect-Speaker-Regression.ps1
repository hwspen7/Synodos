param(
    [string]$MeetingId = "",
    [string]$ApiBase = "http://127.0.0.1:8000",
    [int]$MaxSeconds = 120
)

$ErrorActionPreference = "Stop"
if ($MaxSeconds -lt 5 -or $MaxSeconds -gt 7200) {
    throw "MaxSeconds must be between 5 and 7200."
}

try {
    if (-not $MeetingId) {
        $all = Invoke-RestMethod -Uri "$ApiBase/api/meetings" -Method Get
        $latest = @($all.meetings |
            Where-Object { [int]($_.segments) -gt 0 } |
            Sort-Object updated_at -Descending |
            Select-Object -First 1)
        if ($latest.Count -eq 0) { throw "No saved meetings found." }
        $MeetingId = [string]$latest[0].id
    }

    if ($MeetingId -notmatch '^[A-Za-z0-9_-]{1,64}$') {
        throw "Invalid meeting ID."
    }
    $response = Invoke-RestMethod -Uri "$ApiBase/api/meetings/$MeetingId/segments" -Method Get
}
catch {
    Write-Error "Could not read saved speaker diagnostics. Ensure the backend is running. $($_.Exception.Message)"
    exit 1
}

Write-Host "Meeting ID: $MeetingId"
Write-Host "Speaker evidence report (first $MaxSeconds seconds; no text or voice vectors)"
$items = @($response.segments |
    Where-Object { $_.start_ms -le $MaxSeconds * 1000 } |
    Sort-Object start_ms, id)

if ($items.Count -eq 0) {
    Write-Host "No saved segments in the requested interval."
    exit 0
}

$display = foreach ($s in $items) {
    $start = [TimeSpan]::FromMilliseconds([double]$s.start_ms)
    $source = [string]$s.speaker_diarization_source
    $birth = [string]$s.speaker_birth_decision
    $reason = [string]$s.speaker_evidence_status
    $upstream = [string]$s.speaker_upstream_evidence_status
    $available = if ($null -eq $s.speaker_embedding_available) {
        "unknown"
    } elseif ($s.speaker_embedding_available) { "yes" } else { "no" }
    $speaker = $s.speaker_id
    if ($speaker -gt 0 -and -not $s.speaker_uncertain -and
        $s.speaker_final_resolution_status -eq 'independent_acoustic_assigned' -and
        $s.speaker_continuity_id -gt 0) {
        $speaker = $s.speaker_continuity_id
    }
    [PSCustomObject]@{
        Time = $start.ToString("mm\:ss")
        ID = $s.id
        Speaker = $speaker
        Emb = $available
        Reason = $reason
        Final = [string]$s.speaker_final_resolution_status
        Channel = [string]$s.speaker_acoustic_channel
        Gate = [string]$s.speaker_channel_gate_reason
        Upstream = $upstream
        Birth = $birth
        Source = $source
        Activity = [string]$s.speaker_activity_status
    }
}

$display | Format-Table -AutoSize -Wrap
Write-Host ""
Write-Host "Evidence reason counts:"
$display | Group-Object Reason | Sort-Object Count -Descending |
    Select-Object Count, Name | Format-Table -AutoSize
Write-Host "This report contains no transcript text, raw audio or voiceprint vectors."

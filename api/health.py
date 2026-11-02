"""Health endpoint."""

from fastapi import APIRouter
from core.runtime import settings, models, sessions, meeting_sessions

router = APIRouter()

@router.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "sample_rate": settings.sample_rate,
        "vad_threshold_s": settings.vad_threshold_s,
        "rec_threshold_s": settings.rec_threshold_s,
        "merge_gap_s": settings.merge_gap_s,
        "cluster_merge_thr": settings.cluster_merge_thr,
        "meetings_dir": str(settings.meetings_dir),
        "speakers_dir": str(settings.speakers_dir),
        "hotwords_dir": str(settings.hotwords_dir),
        "min_speaker_audio_s": settings.min_speaker_audio_s,
        "spk_match_threshold": settings.spk_match_threshold,
        "models": models.loaded_state(),
        "active_sessions": len(sessions),
        "active_meetings": len(meeting_sessions),
    }

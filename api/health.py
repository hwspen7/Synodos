"""Health endpoint."""

import os

from speech.config import load_settings as load_speech_settings
from fastapi import APIRouter, HTTPException
from core.runtime import infra
from core.metrics import metrics
from runtime_quality.telemetry import telemetry
from core.runtime import settings, models, sessions, meeting_sessions

router = APIRouter()

@router.get("/api/health")
async def health() -> dict:
    speech = load_speech_settings()
    return {
        "status": "ok",
        "realtime": {
            "source_revision": os.getenv("MEETING_SOURCE_REVISION", ""),
            "asr_engine": speech.asr.engine,
            "asr_model": speech.model,
            "independent_diarization": os.getenv(
                "MEETING_INDEPENDENT_DIARIZATION", "off"
            ).strip().lower(),
        },
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
        "infrastructure": infra.safe_summary(),
    }


@router.get("/api/metrics")
async def operational_metrics() -> dict:
    if not infra.metrics_enabled:
        raise HTTPException(status_code=404, detail="not found")
    report = metrics.snapshot()
    report["phase07"] = telemetry.snapshot()
    return report

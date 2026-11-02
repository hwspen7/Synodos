"""Meeting of Pioneers API entrypoint."""

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.health import router as health_router
from api.meetings import router as meeting_router
from api.speakers import router as speaker_router
from api.hotwords import router as hotword_router
from api.websocket import router as realtime_router
from core.runtime import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
app = FastAPI(title="Meeting of Pioneers", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(meeting_router)
app.include_router(speaker_router)
app.include_router(hotword_router)
app.include_router(realtime_router)


@app.on_event("startup")
async def _preload_models() -> None:
    """Load models before the first request."""
    await asyncio.get_running_loop().run_in_executor(None, models.preload)

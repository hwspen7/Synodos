"""Meeting of Pioneers API entrypoint."""

import asyncio
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from observability.logging_setup import configure_logging, request_id
from observability.metrics import metrics
from api.health import router as health_router
from api.speech_features import router as speech_router
from api.meetings import router as meeting_router
from api.speakers import router as speaker_router
from api.hotwords import router as hotword_router
from api.websocket import router as realtime_router
from core.runtime import models, infra, database

configure_logging(infra.log_level, infra.log_format)
app = FastAPI(title="Meeting of Pioneers", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra.origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])


@app.middleware("http")
async def request_observability(request: Request, call_next):
    rid = request_id(request.headers.get("x-request-id"))
    began = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        metrics.increment("http_unhandled_errors")
        logging.getLogger("api").exception("Unhandled HTTP exception", extra={"request_id": rid})
        raise
    response.headers["X-Request-ID"] = rid
    metrics.increment("http_requests")
    metrics.gauge("http_last_duration_s", time.monotonic() - began)
    return response


app.include_router(health_router)
app.include_router(speech_router)
app.include_router(meeting_router)
app.include_router(speaker_router)
app.include_router(hotword_router)
app.include_router(realtime_router)
# PHASE05_INTELLIGENCE_PATCH: optional AI service routes
app.include_router(__import__('api.intelligence', fromlist=['router']).router)


@app.on_event("startup")
async def _preload_models() -> None:
    """Load models before the first request."""
    await asyncio.get_running_loop().run_in_executor(None, models.preload)


@app.on_event("shutdown")
async def _close_resources() -> None:
    models.close()
    if database is not None:
        await asyncio.to_thread(database.close)

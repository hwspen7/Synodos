"""Phase 05 REST surface. Local-first, job-based and opt-in."""
import asyncio
import re
import uuid
import tempfile
from urllib.parse import unquote
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from intelligence.config import load_config
from intelligence.contracts import ALLOWED_JOBS, ALLOWED_MODES
from intelligence import store, retrieval, media
from speech.supported_languages import translation_target

router=APIRouter(prefix='/api/intelligence',tags=['intelligence'])


def _enabled():
    cfg=load_config()
    if not cfg.enabled:
        raise HTTPException(503,'Intelligence is disabled. Set MEETING_AI_ENABLED=1.')
    if __import__('os').environ.get('MEETING_STORAGE_BACKEND') != 'postgres':
        raise HTTPException(503,'Intelligence requires PostgreSQL storage')
    return cfg


def _uuid(value):
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError,AttributeError) as exc:
        raise HTTPException(400,'Invalid resource ID') from exc


def _meeting_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value):
        raise HTTPException(400,'Invalid meeting ID')
    return value


class JobRequest(BaseModel):
    kind: str
    meeting_id: str | None = None
    payload: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1,max_length=2000)
    limit: int = Field(default=8,ge=1,le=30)
    meeting_ids: list[str] | None = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2,max_length=2000)
    meeting_ids: list[str] | None = None


class ReviewRequest(BaseModel):
    revision: int = Field(ge=1)
    status: str


def _scope(meeting_ids):
    try:
        return retrieval.validate_meeting_ids(meeting_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get('/capabilities')
async def capabilities():
    cfg=load_config()
    return {'enabled':cfg.enabled, 'provider':cfg.provider,'text_model':cfg.text_model,
            'embeddings_enabled':cfg.enable_embeddings,'vision_enabled':cfg.enable_vision,
            'media_audio_import_enabled':cfg.enable_audio_import,'translation':'finalized_segments',
            'worker':'separate_process'}


@router.post('/jobs')
async def submit_job(payload:JobRequest):
    _enabled()
    if payload.kind not in ALLOWED_JOBS:
        raise HTTPException(400,'Invalid task kind')
    if payload.kind in ('live_translate', 'translate'):
        desired = payload.payload.get('target')
        if not isinstance(desired, str) or not desired.strip():
            raise HTTPException(400, 'A supported translation target is required')
        try:
            payload.payload['target'] = translation_target(desired)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if payload.meeting_id is None and payload.kind not in ('ask','ask_evidence','index_media','analyze_media'):
        raise HTTPException(400,'Meeting ID required')
    mid=_meeting_id(payload.meeting_id) if payload.meeting_id else None
    if len(str(payload.payload))>15000:
        raise HTTPException(413,'Task payload too large')
    if payload.kind == 'speaker_refine':
        # Offline speaker model must never compete with the active recorder.
        from core.runtime import meeting_sessions
        from speech.offline_speaker import validate_recording
        from speech.config import load_settings as load_speech_settings
        if mid in meeting_sessions:
            raise HTTPException(409, 'Stop recording before starting speaker refinement')
        filename = payload.payload.get('filename')
        if not isinstance(filename, str):
            raise HTTPException(400, 'Saved WAV filename required')
        try:
            validate_recording(load_speech_settings().recording_root, mid, filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
    if payload.kind == 'speaker_insights':
        from core.runtime import meeting_sessions
        from intelligence.speaker_insights import choose_speaker_turns
        if mid in meeting_sessions:
            raise HTTPException(409, 'Stop recording and save the transcript before individual analysis')
        sid = payload.payload.get('speaker_id')
        rev = payload.payload.get('source_revision')
        if type(sid) is not int or sid <= 0 or type(rev) is not int:
            raise HTTPException(400, 'Speaker ID and transcript revision are required')
        try:
            current_revision, turns = await asyncio.to_thread(store.meeting_segments, mid)
            if current_revision != rev:
                raise HTTPException(409, 'Transcript changed. Refresh speaker analysis first')
            choose_speaker_turns(turns, sid)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        payload.payload = {'speaker_id': sid, 'source_revision': rev}
    if payload.kind == 'context_correction':
        # Context review must never run concurrently with a live transcript.
        from core.runtime import meeting_sessions
        if mid in meeting_sessions:
            raise HTTPException(409, 'Stop recording before reviewing transcript')
        from intelligence.context_correction import selection, ALLOWED_MODES
        mode = payload.payload.get('mode', 'conservative')
        offset = payload.payload.get('offset', 0)
        limit = payload.payload.get('limit', 32)
        if mode not in ALLOWED_MODES or type(offset) is not int or type(limit) is not int:
            raise HTTPException(400, 'Invalid contextual correction options')
        if offset < 0 or not 1 <= limit <= 48:
            raise HTTPException(400, 'Review offset or limit out of bounds')
    if payload.kind in ('ask', 'ask_evidence'):
        scope = _scope(payload.payload.get('meeting_ids'))
        if scope is not None:
            payload.payload['meeting_ids'] = scope
    try:
        if mid:
            await asyncio.to_thread(store.meeting_segments,mid)
        jid=await asyncio.to_thread(store.enqueue,payload.kind,mid,payload.payload)
    except ValueError as exc:
        raise HTTPException(404,str(exc)) from exc
    return {'job_id':jid,'status':'queued'}


@router.get('/jobs/{job_id}')
async def get_job(job_id:str):
    _enabled()
    result=await asyncio.to_thread(store.job_status,_uuid(job_id))
    if not result:
        raise HTTPException(404,'Job not found')
    # Completed quote jobs are historical snapshots. Never display old
    # transcript excerpts as current after a manual correction/deletion.
    if (result['kind'] in ('ask', 'ask_evidence')
            and result['status'] == 'completed'
            and isinstance(result.get('result'), dict)):
        sources = result['result'].get('sources') or []
        if sources and not await asyncio.to_thread(
            store.evidence_revisions_current, sources
        ):
            result['result'] = {
                'status': 'stale_source',
                'answer': 'Source transcript changed. Run this question again.',
                'sources': [],
                'quotes': [],
                'factual_entailment_verified': False,
                'automatic_action_allowed': False,
                'review_required': True,
            }
    return result


@router.post('/jobs/{job_id}/cancel')
async def cancel_job(job_id:str):
    _enabled()
    cancelled=await asyncio.to_thread(store.cancel,_uuid(job_id))
    return {'cancelled':cancelled, 'note':'Only queued tasks can be cancelled'}


@router.get('/meetings/{meeting_id}/overview')
async def overview(meeting_id:str):
    _enabled()
    mid=_meeting_id(meeting_id)
    jobs,arts,actions,decisions=await asyncio.gather(
        asyncio.to_thread(store.list_jobs,mid),
        asyncio.to_thread(store.artifacts,mid),
        asyncio.to_thread(store.review_items,mid,'ai_actions'),
        asyncio.to_thread(store.review_items,mid,'ai_decisions'))
    revision,_=await asyncio.to_thread(store.meeting_segments,mid)
    for artifact in arts:
        artifact['stale'] = artifact.get('source_revision') is not None and artifact['source_revision'] != revision
    return {'jobs':jobs,'artifacts':arts,'actions':actions,'decisions':decisions}


@router.get('/meetings/{meeting_id}/speakers/{speaker_id}/insights')
async def individual_speaker_insights(meeting_id: str, speaker_id: int):
    _enabled()
    mid = _meeting_id(meeting_id)
    from intelligence.speaker_insights import choose_speaker_turns, speaker_source_hash
    try:
        revision, turns = await asyncio.to_thread(store.meeting_segments, mid)
        speaker_name, selected = choose_speaker_turns(turns, speaker_id)
        artifact, active_job = await asyncio.to_thread(
            store.latest_speaker_analysis, mid, speaker_id
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if artifact is not None:
        artifact['stale'] = artifact['source_revision'] != revision
        if (set(artifact['content'].get('segment_ids', [])) != {s['id'] for s in selected}
                or artifact['content'].get('source_fingerprint') != speaker_source_hash(selected)):
            artifact['stale'] = True
    return {
        'speaker_id': speaker_id, 'speaker_name': speaker_name,
        'source_revision': revision, 'segment_ids': [s['id'] for s in selected],
        'artifact': artifact, 'active_job': active_job,
    }


@router.post('/review/{kind}/{item_id}')
async def review(kind:str,item_id:str,payload:ReviewRequest):
    _enabled()
    table={'actions':'ai_actions','decisions':'ai_decisions'}.get(kind)
    if not table:
        raise HTTPException(400,'Unknown review kind')
    try:
        version=await asyncio.to_thread(store.review_update,table,_uuid(item_id),payload.revision,payload.status)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    if version is None:
        raise HTTPException(409,'Review version mismatch or item not found')
    return {'ok':True,'revision':version}


@router.post('/search')
async def search(payload:SearchRequest):
    cfg=_enabled()
    from intelligence.model_gateway import ModelGateway
    gateway=ModelGateway(cfg) if cfg.enable_embeddings else None
    hits=await asyncio.to_thread(
        retrieval.search, payload.query, gateway, payload.limit,
        _scope(payload.meeting_ids)
    )
    return {'results':[dict(x,embedding=None) for x in hits]}


@router.post('/ask')
async def ask(payload:AnswerRequest):
    _enabled()
    data = {'question': payload.question}
    scoped = _scope(payload.meeting_ids)
    if scoped is not None:
        data['meeting_ids'] = scoped
    jid=await asyncio.to_thread(store.enqueue,'ask_evidence',None,data)
    return {'job_id':jid,'status':'queued'}

@router.post('/ask/evidence')
async def ask_evidence(payload: AnswerRequest):
    """Queue a quote-only, source-validated answer; never trust LLM prose."""
    _enabled()
    data = {'question': payload.question}
    scoped = _scope(payload.meeting_ids)
    if scoped is not None:
        data['meeting_ids'] = scoped
    jid = await asyncio.to_thread(
        store.enqueue, 'ask_evidence', None, data
    )
    return {
        'job_id': jid, 'status': 'queued',
        'answer_policy': 'verbatim_source_quotes_only',
    }



@router.get('/meetings/{meeting_id}/translations/{language}')
async def translations(meeting_id:str,language:str):
    _enabled()
    if not re.fullmatch(r'[a-z]{2,3}(-[a-z]{2,4})?',language):
        raise HTTPException(400,'Invalid language code')
    return {'translations':await asyncio.to_thread(store.translations,_meeting_id(meeting_id),language)}


@router.post('/media/upload')
async def upload_media(request:Request):
    """Upload raw file bytes, filename in X-File-Name header. No multipart dependency."""
    cfg=_enabled()
    name=unquote(request.headers.get('X-File-Name',''))
    if Path(name).suffix.lower() not in media.SUPPORTED:
        raise HTTPException(415,'Unsupported file format')
    spool=tempfile.SpooledTemporaryFile(max_size=4*1024*1024)
    total=0
    try:
        async for block in request.stream():
            total+=len(block)
            if total>cfg.max_upload_bytes:
                raise HTTPException(413,'Upload exceeds size limit')
            spool.write(block)
        spool.seek(0)
        try:
            fid,path,digest,size,clean=await asyncio.to_thread(media.store_upload,spool,name,cfg)
            await asyncio.to_thread(store.media_record,fid,clean,request.headers.get('content-type','application/octet-stream'),path,digest,size)
            jid=await asyncio.to_thread(store.enqueue,'analyze_media',None,{'media_id':fid},15)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        except Exception:
            if 'path' in locals():
                path.unlink(missing_ok=True)
            raise
        return {'media_id':fid,'job_id':jid,'size_bytes':size,'name':clean}
    finally:
        spool.close()

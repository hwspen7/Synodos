"""PostgreSQL job lifecycle, artifacts, user review, and retrieval persistence."""
import json
import os
import uuid
from contextlib import contextmanager


def _json(value):
    from psycopg.types.json import Jsonb
    return Jsonb(value)


@contextmanager
def connection():
    url = os.environ.get('DATABASE_URL', '')
    if not url:
        raise RuntimeError('DATABASE_URL is required for the intelligence subsystem')
    import psycopg
    with psycopg.connect(url, connect_timeout=10) as conn:
        yield conn


def _record(row, cols):
    return dict(zip(cols, row)) if row else None


def enqueue(kind, meeting_id=None, payload=None, priority=10):
    from intelligence.contracts import safe_job_kind
    safe_job_kind(kind)
    # Idempotent while the same task is already queued or running. Do not
    # submit duplicate GPU-intensive work from repeated frontend clicks.
    payload = payload or {}
    jid = uuid.uuid4()
    with connection() as conn:
        # Serialize the check/insert across Worker/API processes.
        conn.execute('SELECT pg_advisory_xact_lock(618425071)')
        existing = conn.execute("SELECT id FROM ai_jobs WHERE kind=%s "
                                "AND meeting_id IS NOT DISTINCT FROM %s "
                                "AND payload=%s AND status IN ('queued','running') "
                                "ORDER BY created_at DESC LIMIT 1",
                                (kind, meeting_id, _json(payload))).fetchone()
        if existing:
            return str(existing[0])
        conn.execute('INSERT INTO ai_jobs(id, kind, meeting_id, payload, priority) VALUES (%s,%s,%s,%s,%s)',
                     (jid, kind, meeting_id, _json(payload), int(priority)))
    return str(jid)


def job_status(jid):
    with connection() as conn:
        row = conn.execute('SELECT id, kind, meeting_id, status, attempts, error, result, '
                           'created_at, updated_at FROM ai_jobs WHERE id=%s', (jid,)).fetchone()
    if row is None:
        return None
    out = _record(row, ['id','kind','meeting_id','status','attempts','error','result','created_at','updated_at'])
    out['id'] = str(out['id'])
    return out


def list_jobs(meeting_id, limit=30):
    with connection() as conn:
        rows = conn.execute('SELECT id FROM ai_jobs WHERE meeting_id=%s ORDER BY created_at DESC LIMIT %s',
                            (meeting_id, max(1, min(limit, 100)))).fetchall()
    return [job_status(row[0]) for row in rows]


def cancel(jid):
    with connection() as conn:
        row = conn.execute("UPDATE ai_jobs SET status='cancelled', updated_at=now() "
                           "WHERE id=%s AND status='queued' RETURNING id", (jid,)).fetchone()
    return bool(row)


def claim():
    with connection() as conn:
        with conn.transaction():
            row = conn.execute('''SELECT id, kind, meeting_id, payload, attempts FROM ai_jobs
                  WHERE status='queued' ORDER BY priority, created_at
                  FOR UPDATE SKIP LOCKED LIMIT 1''').fetchone()
            if not row:
                return None
            conn.execute("UPDATE ai_jobs SET status='running', attempts=attempts+1, locked_at=now(), updated_at=now() WHERE id=%s", (row[0],))
    return {'id': str(row[0]), 'kind': row[1], 'meeting_id': row[2], 'payload': row[3], 'attempts': row[4] + 1}


def finish(jid, result):
    with connection() as conn:
        conn.execute("UPDATE ai_jobs SET status='completed', result=%s, error=NULL, completed_at=now(), updated_at=now() WHERE id=%s AND status='running'",
                     (_json(result), jid))


def fail(jid, error):
    msg = str(error)[:300]
    with connection() as conn:
        conn.execute("UPDATE ai_jobs SET status=CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END, error=%s, locked_at=NULL, updated_at=now() WHERE id=%s AND status='running'", (msg, jid))


def touch_job(jid):
    with connection() as conn:
        conn.execute("UPDATE ai_jobs SET locked_at=now() WHERE id=%s AND status='running'", (jid,))


def retry_stale(minutes=45):
    # Worker crash recovery, bounded by max_attempts.
    with connection() as conn:
        conn.execute("""UPDATE ai_jobs SET status=CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END,
          error='Worker lease expired', locked_at=NULL, updated_at=now()
          WHERE status='running' AND locked_at < now() - (%s * interval '1 minute')""", (minutes,))


def meeting_segments(meeting_id):
    with connection() as conn:
        row = conn.execute('SELECT revision, state FROM meeting_records WHERE id=%s', (meeting_id,)).fetchone()
    if not row:
        raise ValueError('Meeting not found')
    revision, state = row
    if not state:
        return int(revision), []
    sr = int(state.get('sample_rate') or 16000)
    if sr < 1:
        raise ValueError('Invalid audio sample rate')
    segments = []
    for s in state.get('segments') or []:
        segments.append({'id': int(s['id']), 'text': str(s.get('text') or ''),
                         'speaker': str(s.get('speaker_name') or ''),
                         'revision': int(s.get('revision', 0)),
                         'manually_corrected': bool(s.get('manually_corrected', False)),
                         'language': str(s.get('language') or 'und'),
                         'start_ms': round(int(s.get('start_sample') or 0) * 1000 / sr),
                         'end_ms': round(int(s.get('end_sample') or 0) * 1000 / sr)})
    return int(revision), segments


def save_artifact(meeting_id, job_id, kind, mode, revision, content, model):
    aid = str(uuid.uuid4())
    with connection() as conn:
        existing = conn.execute('SELECT id FROM ai_artifacts WHERE job_id=%s', (job_id,)).fetchone()
        if existing:
            return str(existing[0])
        conn.execute('''INSERT INTO ai_artifacts(id,meeting_id,job_id,kind,mode,source_revision,content,model_name)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',
                     (aid, meeting_id, job_id, kind, mode, revision, _json(content), model))
        if kind == 'extract':
            for item in content.get('actions', []):
                conn.execute('''INSERT INTO ai_actions(id,meeting_id,artifact_id,title,owner_name,due_date,evidence)
                    VALUES(%s,%s,%s,%s,%s,%s,%s)''',
                    (uuid.uuid4(), meeting_id, aid, item['title'], item.get('owner'), item.get('due_date'), _json(item['source_ids'])))
            for item in content.get('decisions', []):
                conn.execute('''INSERT INTO ai_decisions(id,meeting_id,artifact_id,title,evidence)
                    VALUES(%s,%s,%s,%s,%s)''',
                    (uuid.uuid4(), meeting_id, aid, item['title'], _json(item['source_ids'])))
    return aid


def artifacts(meeting_id, limit=30):
    with connection() as conn:
        rows = conn.execute('''SELECT id,kind,mode,source_revision,content,model_name,created_at FROM ai_artifacts
            WHERE meeting_id=%s ORDER BY created_at DESC LIMIT %s''', (meeting_id, max(1,min(limit,100)))).fetchall()
    return [{'id':str(r[0]),'kind':r[1],'mode':r[2],'source_revision':r[3],
             'content':r[4],'model_name':r[5],'created_at':r[6]} for r in rows]


def review_items(meeting_id, table):
    if table not in ('ai_actions', 'ai_decisions'):
        raise ValueError('Invalid review table')
    if table == 'ai_actions':
        with connection() as conn:
            rows = conn.execute('SELECT id,title,status,evidence,revision,owner_name,due_date FROM ai_actions WHERE meeting_id=%s ORDER BY created_at DESC', (meeting_id,)).fetchall()
        return [{'id':str(r[0]),'title':r[1],'status':r[2],'evidence':r[3],
                 'revision':r[4],'owner':r[5],'due_date':str(r[6]) if r[6] else None} for r in rows]
    with connection() as conn:
        rows = conn.execute('SELECT id,title,status,evidence,revision FROM ai_decisions WHERE meeting_id=%s ORDER BY created_at DESC', (meeting_id,)).fetchall()
    return [{'id':str(r[0]),'title':r[1],'status':r[2],'evidence':r[3],'revision':r[4]} for r in rows]


def review_update(table, item_id, revision, status):
    allowed = {'ai_actions': {'proposed','approved','in_progress','done','rejected'},
               'ai_decisions': {'proposed','confirmed','rejected','superseded'}}
    if status not in allowed.get(table, {}):
        raise ValueError('Invalid review status')
    with connection() as conn:
        row = conn.execute(f'UPDATE {table} SET status=%s,revision=revision+1,updated_at=now() '
                           'WHERE id=%s AND revision=%s RETURNING revision', (status,item_id,revision)).fetchone()
    return row[0] if row else None


def replace_chunks(source_kind, source_id, revision, chunks):
    """Commit source-bound chunks only while the indexed revision is current.

    An asynchronous embedding job may finish after a human transcript edit.
    Lock the same meeting_records row used by PostgresMeetingStore.save_state
    BEFORE deleting any previous index. A stale job must never delete newer
    chunks or make edited text searchable as if it were current.
    """
    if source_kind not in ('meeting', 'media'):
        raise ValueError('Unknown source kind')
    if type(revision) is not int or revision < 0:
        raise ValueError('Invalid source revision')
    if not isinstance(chunks, list) or len(chunks) > 20000:
        raise ValueError('Invalid knowledge chunk batch')
    with connection() as conn:
        if source_kind == 'meeting':
            row = conn.execute(
                'SELECT revision FROM meeting_records WHERE id=%s FOR UPDATE',
                (source_id,),
            ).fetchone()
            if row is None or int(row[0]) != revision:
                raise ValueError(
                    'Saved meeting changed while indexing; requeue on the current revision'
                )
        conn.execute(
            'DELETE FROM ai_knowledge WHERE source_kind=%s AND source_id=%s',
            (source_kind, source_id),
        )
        for i, c in enumerate(chunks):
            conn.execute('''INSERT INTO ai_knowledge(id,source_kind,source_id,source_revision,chunk_index,text_content,evidence,embedding,embedding_model)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                (uuid.uuid4(), source_kind, source_id, revision, i,
                 c['text'], _json(c['evidence']),
                 _json(c['embedding']) if c.get('embedding') is not None else None,
                 c.get('embedding_model')))


def knowledge_candidates(query, limit=250):
    # No extension requirement. Cap rows to avoid unbounded Python work.
    with connection() as conn:
        rows = conn.execute('''SELECT k.source_kind,k.source_id,k.source_revision,k.text_content,k.evidence,k.embedding
             FROM ai_knowledge k
             LEFT JOIN meeting_records m ON k.source_kind='meeting' AND m.id=k.source_id
             WHERE k.source_kind='media' OR (k.source_kind='meeting' AND m.revision=k.source_revision)
             ORDER BY k.updated_at DESC LIMIT %s''', (max(1,min(limit,2000)),)).fetchall()
    return [{'source_kind':r[0],'source_id':r[1],'revision':r[2], 'text':r[3],
             'evidence':r[4], 'embedding':r[5]} for r in rows]



def iter_knowledge_candidates(*, meeting_ids=None, include_embeddings=False,
                              batch_size=256):
    """Yield all current revisions with O(batch_size) memory.

    Unlike the old recent-250 query, a relevant older meeting remains
    retrievable. When embeddings are enabled this is an exact linear scan,
    not a pgvector index. Caller must exhaust or close the iterator.
    """
    if meeting_ids is not None:
        if not isinstance(meeting_ids, list) or not meeting_ids:
            raise ValueError("Invalid meeting scope")
        if any(not isinstance(mid, str) for mid in meeting_ids):
            raise ValueError("Invalid meeting scope")
    if type(batch_size) is not int or not 1 <= batch_size <= 2048:
        raise ValueError("Invalid retrieval batch size")
    embedding_col = "k.embedding" if include_embeddings else "NULL::jsonb"
    conditions = [
        "(k.source_kind='media' OR "
        "(k.source_kind='meeting' AND m.revision=k.source_revision))"
    ]
    params = []
    if meeting_ids is not None:
        conditions.append("k.source_kind='meeting' AND k.source_id=ANY(%s::text[])")
        params.append(meeting_ids)
    sql = (
        "SELECT k.source_kind,k.source_id,k.source_revision,"
        "k.text_content,k.evidence," + embedding_col +
        " FROM ai_knowledge k LEFT JOIN meeting_records m "
        "ON k.source_kind='meeting' AND m.id=k.source_id "
        "WHERE " + " AND ".join(conditions) +
        " ORDER BY k.source_kind,k.source_id,k.chunk_index"
    )
    with connection() as conn:
        with conn.cursor(name="ai_retrieval_" + uuid.uuid4().hex) as cursor:
            cursor.execute(sql, params)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for r in rows:
                    yield {
                        'source_kind': r[0], 'source_id': r[1],
                        'revision': r[2], 'text': r[3],
                        'evidence': r[4], 'embedding': r[5],
                    }


def media_record(file_id, file_name, content_type, path, digest, size):
    with connection() as conn:
        conn.execute('INSERT INTO ai_media(id,file_name,content_type,local_path,sha256,size_bytes) VALUES (%s,%s,%s,%s,%s,%s)',
                     (file_id,file_name,content_type,str(path),digest,size))


def get_media(file_id):
    with connection() as conn:
        row = conn.execute('SELECT id,file_name,content_type,local_path,sha256,size_bytes FROM ai_media WHERE id=%s',(file_id,)).fetchone()
    return {'id':str(row[0]), 'name':row[1], 'content_type':row[2], 'path':row[3], 'sha256':row[4], 'size_bytes':row[5]} if row else None


def save_translation(meeting_id, segment_id, source_text, lang, output, model):
    from intelligence.contracts import source_hash
    digest = source_hash(source_text)
    with connection() as conn:
        conn.execute('''INSERT INTO ai_translations(id,meeting_id,segment_id,source_text,source_hash,target_language,translated_text,model_name)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(meeting_id,segment_id,source_hash,target_language)
          DO UPDATE SET translated_text=EXCLUDED.translated_text,model_name=EXCLUDED.model_name,created_at=now()''',
          (uuid.uuid4(),meeting_id,segment_id,source_text,digest,lang,output,model))


def translations(meeting_id, target):
    from intelligence.contracts import source_hash
    _, segs = meeting_segments(meeting_id)
    by_id = {s['id']:s for s in segs}
    with connection() as conn:
        rows = conn.execute('''SELECT segment_id,source_hash,translated_text FROM ai_translations
             WHERE meeting_id=%s AND target_language=%s''',(meeting_id,target)).fetchall()
    return [{'segment_id':r[0], 'text':r[2]} for r in rows
            if r[0] in by_id and source_hash(by_id[r[0]]['text']) == r[1]]

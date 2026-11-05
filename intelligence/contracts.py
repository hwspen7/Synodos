"""Evidence-constrained structured artifacts and reusable chunking."""
import hashlib
import json
import re
import uuid

ALLOWED_MODES = frozenset({'meeting', 'research', 'classroom', 'interview', 'standup', 'sales'})
ALLOWED_JOBS = frozenset({'summary', 'extract', 'research', 'classroom', 'translate', 'index_meeting', 'index_media', 'ask', 'ask_evidence', 'analyze_media', 'live_translate', 'context_correction', 'speaker_refine'})


def evidence_ids(proposed, valid):
    if not isinstance(proposed, list):
        return []
    return [int(x) for x in proposed if isinstance(x, int) and not isinstance(x, bool) and x in valid][:15]


def safe_artifact(raw, valid_ids):
    """Drop ungrounded decisions/actions rather than promoting invented facts."""
    valid = set(valid_ids)
    summary = str(raw.get('summary') or '')[:20000]
    items = {}
    for field in ('actions', 'decisions', 'open_questions'):
        values = raw.get(field)
        if not isinstance(values, list):
            values = []
        cleaned = []
        for item in values[:40]:
            if not isinstance(item, dict):
                continue
            ev = evidence_ids(item.get('source_ids'), valid)
            title = str(item.get('title') or '').strip()[:600]
            if not title or not ev:
                continue
            entry = {'title': title, 'source_ids': ev}
            if field == 'actions':
                entry['owner'] = str(item.get('owner'))[:120] if item.get('owner') else None
                due = item.get('due_date')
                entry['due_date'] = due if isinstance(due, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', due) else None
            cleaned.append(entry)
        items[field] = cleaned
    return {'summary': summary, **items}


def source_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_chunks(segments, max_chars=1300):
    """Bound every knowledge chunk without truncating long caption segments.

    Preserve source segment ID and timestamps in every piece, even when one
    human-reviewed caption exceeds the embedding model's input window.
    A split piece is still just source text, not a new transcript segment.
    """
    if type(max_chars) is not int or not 8 <= max_chars <= 8000:
        raise ValueError('Knowledge chunk bound must be 8-8000 characters')
    chunks, current, length = [], [], 0
    for seg in segments:
        sid = int(seg['id'])
        remaining = str(seg.get('text') or '').strip()
        if not remaining:
            continue
        while remaining:
            if len(remaining) <= max_chars:
                piece, remaining = remaining, ''
            else:
                # Split English near whitespace when possible, but preserve
                # unspaced languages and IDs without discarding characters.
                boundary = max(remaining.rfind(' ', 0, max_chars + 1),
                               remaining.rfind('\\n', 0, max_chars + 1))
                if boundary < max_chars // 2:
                    boundary = max_chars
                piece = remaining[:boundary].rstrip()
                remaining = remaining[boundary:].lstrip()
            if not piece:
                continue
            if current and length + len(piece) > max_chars:
                chunks.append(_chunk(current))
                current, length = [], 0
            current.append({
                'id': sid, 'text': piece,
                'start_ms': int(seg.get('start_ms', 0)),
                'end_ms': int(seg.get('end_ms', 0)),
            })
            length += len(piece)
    if current:
        chunks.append(_chunk(current))
    return chunks

def _chunk(segs):
    return {'text': '\n'.join(f"[{s['id']}] {s['text']}" for s in segs),
            'evidence': {'segment_ids': [s['id'] for s in segs], 'start_ms': segs[0]['start_ms'],
                         'end_ms': segs[-1]['end_ms']}}


def safe_job_kind(kind):
    if kind not in ALLOWED_JOBS:
        raise ValueError('Unsupported task kind')
    return kind

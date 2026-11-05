"""Revision-aware multilingual retrieval with bounded-memory ranking.

Uses a PostgreSQL server cursor instead of silently discarding older
meetings before relevance ranking. Embedding similarity is an exact
linear scan of JSONB vectors, not an approximate vector index.
"""
import heapq
import math
import re

from intelligence import store


_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did",
    "do", "does", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "our", "the", "their", "there", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with",
})


def validate_meeting_ids(meeting_ids):
    """None means search all locally accessible sources, [] is invalid."""
    if meeting_ids is None:
        return None
    if not isinstance(meeting_ids, list) or not 1 <= len(meeting_ids) <= 20:
        raise ValueError("meeting_ids must contain 1-20 meeting identifiers")
    if any(not isinstance(mid, str) or
           not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", mid) for mid in meeting_ids):
        raise ValueError("Invalid meeting scope")
    return list(dict.fromkeys(meeting_ids))


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        aa = [float(x) for x in a]
        bb = [float(x) for x in b]
        dot = sum(x * y for x, y in zip(aa, bb))
        denom = math.sqrt(sum(x * x for x in aa)) * math.sqrt(
            sum(y * y for y in bb)
        )
        value = dot / denom if denom else 0.0
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def query_terms(query):
    """Include CJK and Hangul bigrams for unspaced meeting transcripts."""
    tokens = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
    result = []
    for token in tokens:
        if any(
            "\u3400" <= c <= "\u9fff" or
            "\u3040" <= c <= "\u30ff" or
            "\uac00" <= c <= "\ud7af"
            for c in token
        ):
            result.extend(
                token[i:i + 2] for i in range(max(1, len(token) - 1))
            )
        elif len(token) > 1 and token not in _STOPWORDS:
            result.append(token)
    return list(dict.fromkeys(result))


def lexical_score(query, text):
    words = query_terms(query)
    corpus = str(text).casefold()
    if not words:
        return float(query.casefold() in corpus)
    return sum(1 for word in words if word in corpus) / len(words)


def rank(query, candidates, embedding=None, limit=8):
    """Exact top-k scoring over an iterable, using O(k) result memory."""
    if type(limit) is not int or not 1 <= limit <= 30:
        raise ValueError("Retrieval limit must be 1-30")
    best = []
    for index, row in enumerate(candidates):
        lexical = lexical_score(query, row["text"])
        semantic = max(0.0, _cosine(embedding, row.get("embedding")))
        if embedding is not None and row.get("embedding"):
            score = 0.35 * lexical + 0.65 * semantic
        else:
            score = lexical
        if score <= 0 or not math.isfinite(score):
            continue
        # Earlier source rows win exact-score ties reproducibly.
        entry = (score, -index, row)
        if len(best) < limit:
            heapq.heappush(best, entry)
        elif entry[:2] > best[0][:2]:
            heapq.heapreplace(best, entry)
    return [dict(row, score=round(score, 4))
            for score, _, row in sorted(best, reverse=True)]


def search(query, gateway=None, limit=8, meeting_ids=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("Search query must contain 1-2000 characters")
    scoped = validate_meeting_ids(meeting_ids)
    embedding = (
        gateway.embed(query)
        if gateway and gateway.cfg.enable_embeddings else None
    )
    return rank(
        query,
        store.iter_knowledge_candidates(
            meeting_ids=scoped, include_embeddings=embedding is not None
        ),
        embedding,
        limit,
    )

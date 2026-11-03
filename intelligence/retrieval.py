"""Multilingual hybrid retrieval without a mandatory pgvector installation."""
import math
import re
from intelligence import store


def _cosine(a,b):
    if not a or not b or len(a)!=len(b):
        return 0.0
    numerator = sum(float(x)*float(y) for x,y in zip(a,b))
    m = math.sqrt(sum(float(x)**2 for x in a)*sum(float(y)**2 for y in b))
    return numerator / m if m else 0.0


def lexical_score(query, text):
    words = [w for w in re.findall(r'[\w]+', query.casefold()) if len(w)>1]
    corpus=text.casefold()
    if not words:
        return float(query.casefold() in corpus)
    hits=sum(1 for word in set(words) if word in corpus)
    # Chinese substrings and English tokens both have coverage.
    return hits / len(set(words))


def rank(query, candidates, embedding=None, limit=8):
    scored = []
    for r in candidates:
        lexical = lexical_score(query, r['text'])
        semantic = max(0.0, _cosine(embedding, r.get('embedding')))
        score = (0.35*lexical + 0.65*semantic) if embedding is not None and r.get('embedding') else lexical
        if score > 0:
            scored.append((score,r))
    scored.sort(key=lambda v:v[0], reverse=True)
    return [dict(r, score=round(s,4)) for s,r in scored[:limit]]


def search(query, gateway=None, limit=8):
    if not query or len(query)>2000:
        raise ValueError('Search query must contain 1-2000 characters')
    embedding = gateway.embed(query) if gateway and gateway.cfg.enable_embeddings else None
    return rank(query,store.knowledge_candidates(query),embedding,limit)

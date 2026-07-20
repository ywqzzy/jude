"""BM25 with GLOBAL corpus statistics — for exact distributed full-text ranking.

Lance's per-shard `full_text_search` scores each shard against its OWN IDF and
average document length, so merging shard-local scores gives a wrong global
ranking (a term rare globally but common in one shard is mis-weighted there).
This module recomputes BM25 with corpus-wide (N, df, avgdl), so distributed FTS
ranks as if the corpus were one index.

Tokenization here is a simple lowercase word split — used consistently for the
document-frequency pre-pass, the candidate term frequencies, and the rescore, so
the scoring is internally consistent (it does not have to match Lance's own
tokenizer, which is only used to SELECT candidates).
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower()) if text else []


def query_terms(query: str) -> list[str]:
    # unique query terms, order-preserving
    seen: dict[str, None] = {}
    for t in tokenize(query):
        seen.setdefault(t, None)
    return list(seen)


def doc_termstats(text: str, terms: list[str]) -> tuple[int, dict[str, int]]:
    """(doc_length_in_tokens, {term: tf}) for the given query terms in one doc."""
    toks = tokenize(text)
    tf: dict[str, int] = {t: 0 for t in terms}
    for w in toks:
        if w in tf:
            tf[w] += 1
    return len(toks), tf


def idf(n_docs: int, df: int) -> float:
    """Robertson/Sparck-Jones IDF with the BM25 +0.5 smoothing, floored at 0."""
    return max(0.0, math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))


def bm25_score(tf: dict[str, int], doc_len: int, idfs: dict[str, float],
               avgdl: float, k1: float = 1.2, b: float = 0.75) -> float:
    """BM25 of one doc given per-term tf, the doc length, global IDFs, and the
    global average document length."""
    if avgdl <= 0:
        return 0.0
    score = 0.0
    denom_len = k1 * (1.0 - b + b * (doc_len / avgdl))
    for term, w in idfs.items():
        f = tf.get(term, 0)
        if f:
            score += w * (f * (k1 + 1.0)) / (f + denom_len)
    return score

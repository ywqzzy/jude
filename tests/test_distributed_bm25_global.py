"""A2: distributed BM25 with GLOBAL IDF ranks as if the corpus were one index.

Merging shard-local BM25 scores mis-ranks a term that is rare globally but
common in one shard. The global-IDF two-pass (corpus-wide N/df/avgdl, then
rescore) must match a single-node BM25 reference computed over the concatenated
corpus with the same tokenizer.
"""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

import jude
from jude import _bm25

lance = pytest.importorskip("lance")
ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _shard(docs, ids):
    p = tempfile.mkdtemp() + "/ds"
    jude._lance.write(pa.table({"id": pa.array(ids, type=pa.int64()), "text": docs}), p, mode="create")
    jude._lance.create_fts_index(p, "text")
    return p


def _reference_bm25(all_docs, all_ids, query, k):
    """Single-node global BM25 over the whole corpus (same tokenizer)."""
    terms = _bm25.query_terms(query)
    n = len(all_docs)
    df = {t: 0 for t in terms}
    total = 0
    per_doc = []
    for txt in all_docs:
        dl, tf = _bm25.doc_termstats(txt, terms)
        total += dl
        per_doc.append((dl, tf))
        for t in terms:
            if tf[t] > 0:
                df[t] += 1
    avgdl = total / n
    idfs = {t: _bm25.idf(n, df[t]) for t in terms}
    scored = [(all_ids[i], _bm25.bm25_score(tf, dl, idfs, avgdl)) for i, (dl, tf) in enumerate(per_doc)]
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return [i for i, _ in scored[:k]]


def test_global_idf_matches_single_node_ranking():
    from jude.runners.ray import RayRunner
    from jude import vector

    # "rare" appears once globally (in shard 1); "common" is everywhere. A
    # shard-local IDF would over-weight "rare" only within its shard.
    s0_docs = ["common common common topic alpha", "common words here about beta",
               "common common filler gamma", "common delta text"]
    s1_docs = ["common rare unusual sentence here", "common common ordinary line",
               "totally common padding", "common common common"]
    p0 = _shard(s0_docs, [0, 1, 2, 3])
    p1 = _shard(s1_docs, [4, 5, 6, 7])
    all_docs = s0_docs + s1_docs
    all_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    r = RayRunner(num_workers=2)
    out = vector.distributed_fts([p0, p1], "text", "common rare", k=5,
                                 columns=["id", "text"], global_idf=True, runner=r)
    got = out.column("id").to_pylist()
    ref = _reference_bm25(all_docs, all_ids, "common rare", 5)
    # top result must match the global reference (doc 4 has the rare term)
    assert got[0] == ref[0] == 4
    # descending global scores
    scores = out.column("_score").to_pylist()
    assert scores == sorted(scores, reverse=True)


def test_global_idf_top1_is_rare_term_doc():
    from jude.runners.ray import RayRunner
    from jude import vector

    p0 = _shard(["the term appears here", "the term is common in shard zero",
                 "the term again", "the term yet again"], [0, 1, 2, 3])
    p1 = _shard(["a special keyword lives alone", "nothing notable", "plain text", "more plain"],
                [4, 5, 6, 7])
    r = RayRunner(num_workers=2)
    out = vector.distributed_fts([p0, p1], "text", "special keyword", k=3, runner=r)
    assert out.column("id").to_pylist()[0] == 4   # the doc with the rare query terms


def test_local_idf_mode_still_works():
    from jude.runners.ray import RayRunner
    from jude import vector

    p0 = _shard(["dogs and cats", "birds"], [0, 1])
    p1 = _shard(["dogs dogs dogs", "fish"], [2, 3])
    r = RayRunner(num_workers=2)
    out = vector.distributed_fts([p0, p1], "text", "dogs", k=2, columns=["id"],
                                 global_idf=False, runner=r)
    assert out.num_rows >= 1
    assert 2 in out.column("id").to_pylist()   # the dogs-heavy doc

"""Vector retrieval correctness: recall metric, MMR diversity, range search, and
resident two-stage ANN recall vs exact brute force. (Fills the audit's blind
spot: the fast vector paths had no correctness tests.)"""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import vector


# --- recall_at_k (pure) ------------------------------------------------------

def test_recall_at_k_perfect_and_partial():
    exact = [1, 2, 3, 4, 5]
    assert vector.recall_at_k(exact, exact) == 1.0
    assert vector.recall_at_k([1, 2, 3, 99, 98], exact) == 0.6   # 3 of 5 correct
    assert vector.recall_at_k([], exact) == 0.0


def test_recall_at_k_truncates_to_k():
    exact = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    approx = [1, 2, 3, 100, 101]
    assert vector.recall_at_k(approx, exact, k=5) == 0.6  # top-5 of exact vs approx


# --- MMR diversity -----------------------------------------------------------

def test_mmr_returns_k_and_prefers_diversity():
    # 3 near-identical vectors near the query + 1 diverse one; MMR with low
    # lambda should include the diverse vector rather than 3 near-duplicates.
    cands = pa.table({
        "id": [1, 2, 3, 4],
        "v": pa.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]],
                      type=pa.list_(pa.float64(), 2)),
    })
    out = vector.mmr(cands, "v", [1.0, 0.0], k=2, lambda_=0.3, metric="cosine")
    ids = out.column("id").to_pylist()
    assert len(ids) == 2
    assert 1 in ids and 4 in ids   # closest + the diverse one


def test_mmr_high_lambda_is_relevance_first():
    cands = pa.table({
        "id": [1, 2, 3],
        "v": pa.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], type=pa.list_(pa.float64(), 2)),
    })
    out = vector.mmr(cands, "v", [1.0, 0.0], k=1, lambda_=1.0, metric="cosine")
    assert out.column("id").to_pylist() == [1]  # pure relevance -> the closest


# --- range search ------------------------------------------------------------

def test_range_search_only_within_radius():
    con = jude.connect()
    t = pa.table({"id": [1, 2, 3],
                  "v": pa.array([[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0]],
                                type=pa.list_(pa.float64(), 2))})
    con.register("vecs", t)
    out = vector.range_search(con, "vecs", "v", [1.0, 0.0], radius=0.2, metric="cosine")
    ids = set(out.column("id").to_pylist())
    assert 1 in ids and 2 in ids   # cosine dist ~0 and ~0.001
    assert 3 not in ids            # opposite direction, dist ~2


# --- resident two-stage ANN recall vs exact ----------------------------------

def _lance_indexed(n=600, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    p = tempfile.mkdtemp() + "/ds"
    t = pa.table({"id": pa.array(list(range(n)), type=pa.int64()),
                  "vec": pa.array(list(vecs), type=pa.list_(pa.float32(), dim))})
    jude._lance.write(t, p, mode="create")
    jude._lance.create_vector_index(p, "vec", index_type="IVF_FLAT", num_partitions=8)
    return p, vecs


def test_resident_ann_high_recall_vs_exact():
    lance = pytest.importorskip("lance")
    p, vecs = _lance_indexed()
    # exact brute-force top-10 (cosine) for a query
    q = vecs[7].astype("float32")
    qn = q / (np.linalg.norm(q) or 1.0)
    vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    exact = list(np.argsort(1.0 - vn @ qn)[:10])
    out = vector.knn_ann_resident(p, "vec", q.tolist(), k=10, overfetch=20,
                                  nprobes=8, metric="cosine")
    got = out.column("id").to_pylist()
    recall = vector.recall_at_k(got, exact, k=10)
    assert recall >= 0.8, f"recall={recall} got={got} exact={exact}"
    assert got[0] == 7  # the query's own vector is nearest

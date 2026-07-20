"""High-recall large-k vector retrieval: recall measurement, exact large-k,
two-stage ANN rerank."""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import vector


def _vecs(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype("float32")


# --- recall metric -----------------------------------------------------------


def test_recall_at_k_basic():
    exact = [1, 2, 3, 4, 5]
    assert vector.recall_at_k(exact, exact) == 1.0
    assert vector.recall_at_k([1, 2, 3, 9, 8], exact, k=5) == 0.6  # 3 of 5
    assert vector.recall_at_k([], exact) == 0.0
    assert vector.recall_at_k([9, 8], []) == 1.0  # empty ground truth


# --- exact large-k = 100% recall (distributed == single-node) ----------------


def test_distributed_exact_is_full_recall_large_k():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner

    n, d, k = 5000, 16, 2000  # "large k" relative to n
    vecs = _vecs(n, d)
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    q = _vecs(1, d, seed=99)[0].tolist()
    con = jude.connect()
    con.register("emb", t)
    exact_ids = vector.knn(con, "emb", "v", q, k=k, metric="cosine").column("id").to_pylist()
    dist_ids = vector.distributed_knn(t, "v", q, k=k, metric="cosine",
                                      runner=RayRunner(num_workers=4)).column("id").to_pylist()
    # distributed exact == single-node exact -> recall 1.0
    assert vector.recall_at_k(dist_ids, exact_ids, k=k) == 1.0
    assert dist_ids == exact_ids  # exact same order too


def test_high_recall_knn_wrapper():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner

    n, d, k = 3000, 8, 1000
    vecs = _vecs(n, d, seed=3)
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    q = _vecs(1, d, seed=7)[0].tolist()
    con = jude.connect()
    con.register("emb", t)
    exact = vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
    got = vector.high_recall_knn(t, "v", q, k=k, runner=RayRunner(num_workers=4)).column("id").to_pylist()
    assert vector.recall_at_k(got, exact, k=k) == 1.0  # exact path


# --- two-stage ANN rerank over Lance -----------------------------------------


def test_knn_rerank_high_recall():
    lance = pytest.importorskip("lance")
    n, d, k = 20000, 32, 50
    vecs = _vecs(n, d, seed=5)
    path = tempfile.mkdtemp(prefix="jude_hr_") + "/ds"
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    jude._lance.write(t, path, mode="create")
    con = jude.connect()
    con.register("emb", t)
    # build an ANN index
    con.create_lance_vector_index(path, "v", index_type="IVF_PQ", metric="cosine",
                                  num_partitions=64, num_sub_vectors=8)
    q = _vecs(1, d, seed=123)[0].tolist()
    exact = vector.knn(con, "emb", "v", q, k=k, metric="cosine").column("id").to_pylist()

    # plain ANN (small nprobes) vs two-stage rerank (over-fetch + exact rerank).
    # Recall scales with over-fetch + nprobes; use generous settings to show the
    # technique reaching high recall.
    plain = jude._lance.vector_search(path, "v", q, k=k, nprobes=1).column("id").to_pylist()
    reranked = vector.knn_rerank(path, "v", q, k=k, overfetch=40, nprobes=64, metric="cosine").column("id").to_pylist()

    r_plain = vector.recall_at_k(plain, exact, k=k)
    r_rerank = vector.recall_at_k(reranked, exact, k=k)
    # two-stage rerank recalls at least as well as plain ANN (the core claim);
    # with generous over-fetch it reaches high recall (bar kept honest for a
    # small PQ index — recall rises further with more over-fetch / IVF_FLAT).
    assert r_rerank >= r_plain
    assert r_rerank >= 0.6

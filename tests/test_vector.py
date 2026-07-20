"""In-process vector search over DuckDB's native array functions + VSS HNSW."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import jude
from jude import vector


def _emb_table(con):
    # 5 unit-ish vectors in 3D
    vecs = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ]
    con.execute("CREATE TABLE emb (id INTEGER, v FLOAT[3])")
    for i, vv in enumerate(vecs):
        con.execute(f"INSERT INTO emb VALUES ({i}, {vv})")
    return con


def test_knn_cosine():
    con = jude.connect()
    _emb_table(con)
    out = vector.knn(con, "emb", "v", [1.0, 0.0, 0.0], k=2, metric="cosine")
    ids = out.column("id").to_pylist()
    assert ids[0] == 0  # exact match nearest
    assert ids[1] == 1  # next closest
    assert "_distance" in out.column_names
    assert out.column("_distance").to_pylist()[0] < 1e-6


def test_knn_l2():
    con = jude.connect()
    _emb_table(con)
    out = vector.knn(con, "emb", "v", [0.0, 0.0, 1.0], k=1, metric="l2")
    assert out.column("id").to_pylist() == [3]


def test_add_similarity_cosine():
    con = jude.connect()
    _emb_table(con)
    out = vector.add_similarity(con, "emb", "v", [1.0, 0.0, 0.0], metric="cosine")
    sims = dict(zip(out.column("id").to_pylist(), out.column("similarity").to_pylist()))
    assert sims[0] == pytest.approx(1.0, abs=1e-5)  # identical
    assert sims[4] == pytest.approx(-1.0, abs=1e-5)  # opposite
    assert sims[2] == pytest.approx(0.0, abs=1e-5)  # orthogonal


def test_knn_with_where():
    con = jude.connect()
    _emb_table(con)
    out = vector.knn(con, "emb", "v", [1.0, 0.0, 0.0], k=5, metric="cosine", where="id != 0")
    assert 0 not in out.column("id").to_pylist()


def test_hnsw_index_and_search():
    con = jude.connect()
    _emb_table(con)
    try:
        vector.create_hnsw_index(con, "emb", "v", metric="cosine")
    except Exception as e:  # noqa: BLE001 — vss may be unavailable offline
        pytest.skip(f"vss extension unavailable: {e}")
    # after the index, KNN via the same distance fn should still be correct
    out = vector.knn(con, "emb", "v", [1.0, 0.0, 0.0], k=1, metric="cosine")
    assert out.column("id").to_pylist() == [0]


def test_distributed_knn_matches_local():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    import pyarrow as pa
    from jude.runners.ray import RayRunner

    # 200 random 8-d vectors; distributed top-5 == single-node top-5
    import random

    random.seed(0)
    vecs = [[random.random() for _ in range(8)] for _ in range(200)]
    t = pa.table({"id": list(range(200)), "v": pa.array(vecs, type=pa.list_(pa.float32(), 8))})
    q = [0.5] * 8
    con = jude.connect()
    con.register("emb", t)
    local = vector.knn(con, "emb", "v", q, k=5, metric="cosine").column("id").to_pylist()
    dist = vector.distributed_knn(t, "v", q, k=5, metric="cosine", runner=RayRunner(num_workers=4))
    assert dist.column("id").to_pylist() == local

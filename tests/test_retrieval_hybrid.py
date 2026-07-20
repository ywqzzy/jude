"""Hybrid analytical retrieval: retrieval fused with DuckDB analytics
(jude.retrieval — P0 of the DuckDB distributed-retrieval design)."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import retrieval, vector

lance = pytest.importorskip("lance")


def _clustered(n, d, clusters=16, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    return v, c


def _tbl(v, ids, extra=None):
    child = pa.array(v.reshape(-1), type=pa.float32())
    cols = {"id": pa.array(ids, type=pa.int64()),
            "v": pa.FixedSizeListArray.from_arrays(child, v.shape[1])}
    if extra:
        cols.update(extra)
    return pa.table(cols)


def test_search_then_sql_joins_and_aggregates():
    con = jude.connect()
    # a "users" dimension table lives in DuckDB
    con.register("users", pa.table({"id": list(range(10)),
                                    "org": ["a", "b"] * 5}))
    # candidate hits from "retrieval" (here a literal table with a score)
    hits = pa.table({"id": [1, 2, 3, 4], "_distance": [0.1, 0.2, 0.3, 0.4]})
    out = retrieval.search_then_sql(
        con,
        """SELECT u.org, count(*) AS n, avg(h._distance) AS rel
           FROM hits h JOIN users u ON u.id = h.id
           GROUP BY u.org ORDER BY u.org""",
        candidates={"hits": hits},
    )
    d = out.to_pydict()
    assert d["org"] == ["a", "b"]
    assert d["n"] == [2, 2]  # ids 2,4 -> org b ; 1,3 -> org a
    # re-registering the same name overwrites cleanly (repeatable calls)
    out2 = retrieval.search_then_sql(
        con, "SELECT count(*) c FROM hits",
        candidates={"hits": pa.table({"id": [9]})})
    assert out2.to_pydict() == {"c": [1]}


def test_search_then_sql_lazy_callable():
    con = jude.connect()
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return pa.table({"id": [1, 2], "s": [0.5, 0.9]})

    out = retrieval.search_then_sql(
        con, "SELECT count(*) c, max(s) m FROM cand", candidates={"cand": make})
    assert out.to_pydict() == {"c": [2], "m": [0.9]}
    assert calls["n"] == 1  # evaluated exactly once, lazily


def test_hybrid_analytical_vector_then_filter_aggregate():
    n, d = 20000, 32
    v, c = _clustered(n, d)
    cat = (np.arange(n) % 4)
    path = tempfile.mkdtemp(prefix="jude_hy_") + "/ds"
    jude._lance.write(_tbl(v, np.arange(n), {"cat": pa.array(cat.tolist())}), path, mode="create")
    jude.connect().create_lance_vector_index(path, "v", index_type="IVF_FLAT",
                                              metric="cosine", num_partitions=64)
    con = jude.connect()
    q = c[2].tolist()
    # retrieve top-500 similar, then group-by category with avg distance — all in SQL
    out = retrieval.hybrid_analytical(
        con, path,
        """SELECT cat, count(*) n, avg(_distance) rel
           FROM hits WHERE cat IN (0, 2) GROUP BY cat ORDER BY cat""",
        vector_query=q, vector_column="v", k=500, nprobes=16,
    )
    d = out.to_pydict()
    assert set(d["cat"]).issubset({0, 2})   # WHERE filter applied
    assert sum(d["n"]) > 0                  # got candidates through the analytics

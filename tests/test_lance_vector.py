"""Lance vector search + secondary indexing — fills the ANN gap neither stock
DuckDB nor Vane has natively. ANN results are jude relations, so they compose
with SQL (hybrid search = ANN + predicate pushdown). Gated on pylance."""
import os
import tempfile

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("lance")
import jude


def _dataset(n=512, dim=8):
    d = os.path.join(tempfile.mkdtemp(), "v.lance")
    c = jude.connect()
    vecs = [np.random.rand(dim).astype("float32") for _ in range(n)]
    t = pa.table({
        "id": list(range(n)),
        "cat": [i % 3 for i in range(n)],
        "emb": pa.array(vecs, type=pa.list_(pa.float32(), dim)),
    })
    c.from_arrow(t).write_lance(d, mode="create")
    return c, d, vecs


def test_vector_search_finds_nearest():
    c, d, vecs = _dataset()
    c.create_lance_vector_index(d, "emb", num_partitions=4, num_sub_vectors=2)
    res = c.lance_vector_search(d, "emb", list(vecs[0]), k=5)
    assert "_distance" in res.columns
    assert len(res.fetchall()) == 5
    # the query vector itself is the closest row
    assert res.order("_distance").fetchone()[0] == 0


def test_hybrid_search_filter_pushdown():
    c, d, vecs = _dataset()
    c.create_lance_vector_index(d, "emb", num_partitions=4, num_sub_vectors=2)
    res = c.lance_vector_search(d, "emb", list(vecs[0]), k=5, filter="cat = 1")
    assert all(row[1] == 1 for row in res.fetchall())


def test_ann_composes_with_sql():
    c, d, vecs = _dataset()
    c.create_lance_vector_index(d, "emb", num_partitions=4, num_sub_vectors=2)
    # ANN result -> ordinary SQL aggregate over the top-k
    top = c.lance_vector_search(d, "emb", list(vecs[0]), k=10)
    assert top.aggregate("count(*)").fetchone()[0] == 10


def test_scalar_index_then_read():
    c, d, _ = _dataset()
    c.create_lance_scalar_index(d, "cat", index_type="BITMAP")
    assert len(jude.read_lance(d, filter="cat = 2").fetchall()) > 0

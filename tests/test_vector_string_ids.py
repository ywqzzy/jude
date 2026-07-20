"""A4: vector search must handle non-integer ids (string / UUID) and honor a
configurable id_column instead of hardcoding int64. Before this fix the resident
and distributed paths coerced ids via int(x) and typed output arrays as int64, so
string/UUID doc-ids either crashed or were silently corrupted.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import _lance

lance = pytest.importorskip("lance")


def _str_id_tbl(n=200, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    return pa.table({"doc_id": pa.array([f"doc-{i:05d}" for i in range(n)]),
                     "vec": pa.array(list(vecs), type=pa.list_(pa.float32(), dim))})


def _exact_topk(tbl, q, k, id_col="doc_id"):
    v = np.asarray(tbl.column("vec").to_pylist(), dtype="float32")
    qn = np.linalg.norm(q) or 1.0
    d = 1.0 - (v @ q) / (np.linalg.norm(v, axis=1) * qn)
    order = np.argsort(d)[:k]
    return [tbl.column(id_col)[int(i)].as_py() for i in order]


def test_resident_ann_string_ids():
    from jude.vector import knn_ann_resident

    p = tempfile.mkdtemp() + "/ds"
    t = _str_id_tbl()
    _lance.write(t, p, mode="create")
    _lance.create_vector_index(p, "vec", index_type="IVF_FLAT", num_partitions=4)
    q = t.column("vec")[3].as_py()
    out = knn_ann_resident(p, "vec", q, k=5, overfetch=10, metric="cosine",
                           id_column="doc_id")
    got = out.column("id").to_pylist()
    assert all(isinstance(x, str) for x in got)      # ids stay strings
    assert out.column("id").type == pa.string()      # not silently int64
    assert "doc-00003" in got                         # the query's own row


def test_distributed_resident_string_ids():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner
    from jude.vector import distributed_knn_resident

    # two shards, string ids
    paths = []
    tbls = []
    for s in range(2):
        p = tempfile.mkdtemp() + f"/shard{s}"
        t = pa.table({"doc_id": pa.array([f"s{s}-{i:04d}" for i in range(150)]),
                      "vec": pa.array(list(np.random.default_rng(s).standard_normal((150, 16)).astype("float32")),
                                      type=pa.list_(pa.float32(), 16))})
        _lance.write(t, p, mode="create")
        paths.append(p)
        tbls.append(t)
    r = RayRunner(num_workers=2)
    q = tbls[0].column("vec")[0].as_py()
    out = distributed_knn_resident(paths, "vec", q, k=5, metric="l2", id_column="doc_id",
                                   runner=r)
    got = out.column("id").to_pylist()
    assert out.column("id").type == pa.string()
    assert all(isinstance(x, str) for x in got)
    assert "s0-0000" in got  # the query vector's own shard-0 id is the nearest


def test_distributed_resident_batch_string_ids():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner
    from jude.vector import distributed_knn_resident_batch

    p = tempfile.mkdtemp() + "/shard"
    t = pa.table({"doc_id": pa.array([f"u-{i:04d}" for i in range(120)]),
                  "vec": pa.array(list(np.random.default_rng(1).standard_normal((120, 16)).astype("float32")),
                                  type=pa.list_(pa.float32(), 16))})
    _lance.write(t, p, mode="create")
    r = RayRunner(num_workers=2)
    qs = [t.column("vec")[0].as_py(), t.column("vec")[7].as_py()]
    outs = distributed_knn_resident_batch([p], "vec", qs, k=3, metric="l2",
                                          id_column="doc_id", runner=r)
    assert len(outs) == 2
    for o in outs:
        assert o.column("id").type == pa.string()
    assert "u-0000" in outs[0].column("id").to_pylist()
    assert "u-0007" in outs[1].column("id").to_pylist()

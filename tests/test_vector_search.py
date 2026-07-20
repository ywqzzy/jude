"""VectorSearch encapsulation: exact (single/distributed) + ANN + recall."""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import VectorSearch


def _clustered(n, d, seed=0, clusters=20):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    vecs = (centers[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    q = (centers[1] + 0.15 * rng.standard_normal(d)).astype("float32").tolist()
    return vecs, q


def _table(vecs, d):
    return pa.table({"id": list(range(len(vecs))), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})


def test_exact_single_node():
    vecs, q = _clustered(2000, 16)
    vs = VectorSearch(_table(vecs, 16), column="v", metric="cosine")
    out = vs.search(q, k=50)
    assert out.num_rows == 50
    assert "_distance" in out.column_names
    # nearest-first
    ds = out.column("_distance").to_pylist()
    assert ds == sorted(ds)


def test_exact_full_recall():
    vecs, q = _clustered(3000, 16, seed=2)
    vs = VectorSearch(_table(vecs, 16), column="v")
    assert vs.recall_vs_exact(q, k=100) == 1.0  # exact vs exact


def test_exact_distributed_matches_single():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner

    vecs, q = _clustered(4000, 16, seed=3)
    t = _table(vecs, 16)
    single = VectorSearch(t, column="v").search(q, k=200).column("id").to_pylist()
    dist = VectorSearch(t, column="v", distributed=True, runner=RayRunner(num_workers=4)).search(q, k=200).column("id").to_pylist()
    assert dist == single  # distributed exact == single-node exact


def test_batch_search():
    vecs, q = _clustered(1000, 8)
    vs = VectorSearch(_table(vecs, 8), column="v")
    res = vs.search_batch([q, q, q], k=10)
    assert len(res) == 3
    assert all(r.num_rows == 10 for r in res)


def test_ann_strategy_high_recall():
    lance = pytest.importorskip("lance")
    vecs, q = _clustered(20000, 32, seed=5)
    path = tempfile.mkdtemp(prefix="jude_vs_") + "/ds"
    jude._lance.write(_table(vecs, 32), path, mode="create")
    vs = VectorSearch(path, column="v", strategy="ann", overfetch=5, nprobes=50)
    vs.build_index(index_type="IVF_FLAT", num_partitions=100)
    # clustered data + IVF_FLAT + rerank -> high recall
    r = vs.recall_vs_exact(q, k=100)
    assert r >= 0.9


def test_ann_requires_path():
    with pytest.raises(ValueError):
        VectorSearch(pa.table({"v": [[1.0, 2.0]]}), strategy="ann")

"""Distributed Lance vector-index build: centroids trained across workers."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import vector

ray = pytest.importorskip("ray")
lance = pytest.importorskip("lance")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def test_distributed_index_build_is_queryable():
    from jude.runners.ray import RayRunner

    n, d = 40_000, 32
    rng = np.random.default_rng(0)
    c = rng.standard_normal((40, d)).astype("float32")
    lab = rng.integers(0, 40, n)
    vecs = (c[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    path = tempfile.mkdtemp(prefix="jude_distidx_") + "/ds"
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    jude._lance.write(t, path, mode="create")

    r = RayRunner(num_workers=4)
    info = r.distributed_create_vector_index(path, "v", index_type="IVF_FLAT", metric="cosine",
                                             num_partitions=64)
    assert info["column"] == "v"
    assert info["centroids"] in ("distributed-kmeans", "single-node-fallback")

    # the built index must be queryable with high recall vs exact
    con = jude.connect()
    con.register("emb", t)
    q = (c[3] + 0.15 * rng.standard_normal(d)).astype("float32").tolist()
    exact = vector.knn(con, "emb", "v", q, k=50).column("id").to_pylist()
    got = vector.knn_rerank(path, "v", q, k=50, overfetch=5, nprobes=64).column("id").to_pylist()
    assert vector.recall_at_k(got, exact, 50) >= 0.9

"""Billion-scale algorithm at runnable scale: distributed sharded ANN.

Each shard is a pre-indexed Lance dataset; query fans out, merges to global
top-k. Verifies recall vs global exact ground truth. (Runs at ~200k vectors /
4 shards — the same algorithm that scales to 1B across many machines.)"""

from __future__ import annotations

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


def test_distributed_sharded_ann_recall():
    from jude.runners.ray import RayRunner

    d = 32
    n_shards = 4
    per = 50_000  # 200k total
    k = 500
    clusters = 40

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((clusters, d)).astype("float32")

    # build the global table (for exact ground truth) + per-shard indexed datasets
    all_vecs = []
    shard_paths = []
    gid = 0
    for s in range(n_shards):
        lab = rng.integers(0, clusters, per)
        vecs = (centers[lab] + 0.15 * rng.standard_normal((per, d))).astype("float32")
        ids = list(range(gid, gid + per))
        gid += per
        all_vecs.append((ids, vecs))
        path = tempfile.mkdtemp(prefix=f"jude_shard{s}_") + "/ds"
        t = pa.table({"id": ids, "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
        jude._lance.write(t, path, mode="create")
        con = jude.connect()
        con.create_lance_vector_index(path, "v", index_type="IVF_FLAT", metric="cosine",
                                      num_partitions=int(per ** 0.5))
        shard_paths.append(path)

    # global exact ground truth over the union
    gids = [i for ids, _ in all_vecs for i in ids]
    gvecs = np.concatenate([v for _, v in all_vecs])
    gt_tbl = pa.table({"id": gids, "v": pa.array(gvecs.tolist(), type=pa.list_(pa.float32(), d))})
    con = jude.connect()
    con.register("g", gt_tbl)
    q = (centers[2] + 0.15 * rng.standard_normal(d)).astype("float32").tolist()
    exact = vector.knn(con, "g", "v", q, k=k, metric="cosine").column("id").to_pylist()

    # distributed sharded ANN: fan out to shard indexes, merge global top-k
    got = vector.distributed_ann_knn(
        shard_paths, "v", q, k=k, overfetch=5, nprobes=int(per ** 0.5),
        metric="cosine", runner=RayRunner(num_workers=4),
    )
    got_ids = got.column("id").to_pylist()
    assert got.num_rows == k
    # nearest-first
    ds = got.column("_distance").to_pylist()
    assert ds == sorted(ds)
    # recall vs global exact — clustered data + IVF_FLAT + rerank should be high
    r = vector.recall_at_k(got_ids, exact, k=k)
    assert r >= 0.9, f"sharded ANN recall too low: {r}"

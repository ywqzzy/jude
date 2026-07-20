"""Vector search use-case coverage: filtered ANN, range search, MMR, and
cluster-routed distributed ANN. Correctness against exact ground truth."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import vector

lance = pytest.importorskip("lance")


def _clustered(n, d, clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.1 * rng.standard_normal((n, d))).astype("float32")
    return v, c, lab


def _tbl(v, ids, extra=None):
    child = pa.array(v.reshape(-1), type=pa.float32())
    cols = {"id": pa.array(ids, type=pa.int64()),
            "v": pa.FixedSizeListArray.from_arrays(child, v.shape[1])}
    if extra:
        cols.update(extra)
    return pa.table(cols)


def test_range_search_threshold():
    n, d = 5000, 32
    v, c, _ = _clustered(n, d)
    con = jude.connect()
    con.register("emb", _tbl(v, np.arange(n)))
    q = c[3].tolist()
    out = vector.range_search(con, "emb", "v", q, radius=0.15, metric="cosine")
    dists = out.column("_distance").to_pylist()
    # every returned row must be within the radius, and sorted ascending
    assert all(x <= 0.15 + 1e-6 for x in dists)
    assert dists == sorted(dists)
    # radius 0 (or tiny) returns far fewer than a big radius
    big = vector.range_search(con, "emb", "v", q, radius=0.5, metric="cosine").num_rows
    assert big >= out.num_rows


def test_mmr_diversifies():
    d = 16
    # 3 tight clusters; candidates drawn from all — MMR should spread across them
    rng = np.random.default_rng(1)
    cents = rng.standard_normal((3, d)).astype("float32")
    vecs = np.vstack([cents[i] + 0.01 * rng.standard_normal((10, d)) for i in range(3)]).astype("float32")
    labels = np.repeat(np.arange(3), 10)
    cand = _tbl(vecs, np.arange(30), {"lab": pa.array(labels.tolist())})
    q = cents[0].tolist()
    picked = vector.mmr(cand, "v", q, k=3, lambda_=0.3)  # diversity-leaning
    labs = set(picked.column("lab").to_pylist())
    assert len(labs) >= 2  # not all from the query's own cluster
    # pure relevance (lambda=1) should favor the query cluster
    rel = vector.mmr(cand, "v", q, k=3, lambda_=1.0)
    assert 0 in rel.column("lab").to_pylist()


def test_filtered_ann_prefilter():
    n, d = 40000, 32
    v, c, lab = _clustered(n, d)
    cat = (np.arange(n) % 5)  # 5 categories
    path = tempfile.mkdtemp(prefix="jude_fann_") + "/ds"
    jude._lance.write(_tbl(v, np.arange(n), {"cat": pa.array(cat.tolist())}), path, mode="create")
    jude.connect().create_lance_vector_index(path, "v", index_type="IVF_FLAT",
                                              metric="cosine", num_partitions=64)
    q = c[2].tolist()
    got = vector.knn_rerank(path, "v", q, k=50, overfetch=5, nprobes=16,
                            metric="cosine", columns=["id", "cat"], where="cat = 3")
    cats = got.column("cat").to_pylist()
    assert cats and all(x == 3 for x in cats)  # every hit satisfies the predicate


@pytest.mark.skipif("not pytest.importorskip('ray')")
def test_routed_distributed_ann():
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner

    d, per = 32, 8000
    rng = np.random.default_rng(2)
    cents = rng.standard_normal((4, d)).astype("float32") * 3.0  # well-separated clusters
    shard_paths, shard_cents = [], []
    all_v, all_id = [], []
    for i in range(4):
        # 0.35 noise -> points are DISTINCT within a cluster (a well-defined KNN),
        # not near-duplicates (which would make exact top-20 arbitrary).
        sv = (cents[i] + 0.35 * rng.standard_normal((per, d))).astype("float32")
        ids = np.arange(i * per, (i + 1) * per)
        p = tempfile.mkdtemp(prefix=f"jude_route{i}_") + "/ds"
        jude._lance.write(_tbl(sv, ids), p, mode="create")
        jude.connect().create_lance_vector_index(p, "v", index_type="IVF_FLAT",
                                                  metric="cosine", num_partitions=32)
        shard_paths.append(p)
        shard_cents.append(sv.mean(axis=0).tolist())
        all_v.append(sv)
        all_id.append(ids)
    runner = RayRunner(num_workers=4)
    q = (cents[1] + 0.1 * rng.standard_normal(d)).astype("float32")
    # exact ground truth over the whole corpus
    V = np.vstack(all_v)
    ids = np.concatenate(all_id)
    qn = V @ q / (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) or 1.0))
    exact = ids[np.argsort(1 - qn)[:20]].tolist()

    def routed(nsp):
        # nprobes=32 (== num_partitions) makes the INTRA-shard search near-exact,
        # so this test isolates the ROUTING recall (n_shards_probe), not the
        # per-shard IVF approximation.
        got = vector.distributed_ann_knn_routed(shard_paths, shard_cents, "v", q.tolist(),
                                                k=20, n_shards_probe=nsp, overfetch=10,
                                                nprobes=32, runner=runner)
        return vector.recall_at_k(got.column("id").to_pylist(), exact, 20)

    # With well-separated clusters, the query's true neighbors live in its own
    # cluster's shard, so centroid-routing to even 1 shard recovers most of them,
    # and probing more shards is monotonic non-decreasing (never worse).
    r1, r2, r4 = routed(1), routed(2), routed(4)
    assert r4 >= r2 >= r1 - 1e-9
    assert r1 >= 0.9   # routing hit the right cluster
    assert r4 >= 0.95  # probing all shards ~ exhaustive sharded ANN


@pytest.mark.skipif("not pytest.importorskip('ray')")
def test_distributed_fts_and_hybrid():
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner

    d = 16
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "distributed systems use consensus protocols for consistency",
        "vector search finds nearest neighbors in embedding space",
        "the lazy dog sleeps while the quick fox runs",
        "full text search ranks documents by keyword relevance bm25",
        "neural networks learn representations from data",
    ]
    rng = np.random.default_rng(3)
    # 2 shards, FTS + vector index on each
    shard_paths = []
    for s in range(2):
        rows = list(range(s * 3, s * 3 + 3))
        v = rng.standard_normal((3, d)).astype("float32")
        t = _tbl(v, np.array(rows), {"text": pa.array([docs[r] for r in rows])})
        p = tempfile.mkdtemp(prefix=f"jude_fts{s}_") + "/ds"
        jude._lance.write(t, p, mode="create")
        jude._lance.create_fts_index(p, "text")
        jude.connect().create_lance_vector_index(p, "v", index_type="IVF_FLAT",
                                                  metric="cosine", num_partitions=2)
        shard_paths.append(p)
    runner = RayRunner(num_workers=2)

    # distributed BM25: "lazy dog" hits docs 0 and 3 (across both shards)
    got = vector.distributed_fts(shard_paths, "text", "lazy dog", k=5, runner=runner)
    hit_ids = set(got.column("id").to_pylist())
    assert 0 in hit_ids and 3 in hit_ids  # both "lazy dog" docs found across shards

    # distributed hybrid: fuse BM25 + vector, returns fused rows with _rrf_score
    qv = rng.standard_normal(d).astype("float32").tolist()
    fused = vector.distributed_hybrid(shard_paths, "text", "v", "vector search", qv,
                                      k=4, overfetch=5, nprobes=2, runner=runner)
    assert fused.num_rows > 0 and "_rrf_score" in fused.column_names


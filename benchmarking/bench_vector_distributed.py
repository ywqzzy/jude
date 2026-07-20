#!/usr/bin/env python3
"""Fair distributed-vs-single-node vector search benchmark.

Compares, over many queries (latency ms/query + QPS):
  1. single-node EXACT, 1 core (DuckDB threads=1) — the honest 1-core baseline
  2. single-node Lance ANN (IVF_FLAT), 1 dataset
  3. distributed RESIDENT exact, S workers — data pre-partitioned onto workers,
     only the query travels (the CORRECT distributed architecture; contrast with
     distributed_knn which re-ships the table per query)
  4. distributed_ann_knn, S pre-indexed shards — sharded ANN (billion-scale path)

All exact methods are 100% recall; ANN methods report recall vs exact. This
shows where distribution beats 1-core and where sharded ANN wins.

    python benchmarking/bench_vector_distributed.py --n 1000000 --dim 96 --k 100 --shards 4
"""

from __future__ import annotations

import argparse
import math
import statistics
import tempfile
import time

import numpy as np
import pyarrow as pa


def make(n, d, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((200, d)).astype("float32")
    lab = rng.integers(0, 200, n)
    v = (c[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    q = [(c[i % 200] + 0.15 * rng.standard_normal(d)).astype("float32").tolist() for i in range(30)]
    return v, q


def _bench(fn, queries):
    fn(queries[0])
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        lat.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(lat), 1000 / statistics.mean(lat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    import jude
    from jude import vector

    from _bench_ray import connect_ray
    connect_ray(num_cpus=args.shards)
    from jude.runners.ray import RayRunner
    runner = RayRunner(num_workers=args.shards)

    n, d, k, S = args.n, args.dim, args.k, args.shards
    print(f"\nvector search: single-node (1 core) vs distributed ({S} workers)")
    print(f"N={n:,} dim={d} k={k} shards={S}, 30 queries")
    v, queries = make(n, d)
    t = pa.table({"id": list(range(n)), "v": pa.array(v.tolist(), type=pa.list_(pa.float32(), d))})

    # whole-corpus Lance dataset (for single-node ANN) + per-shard datasets
    whole = tempfile.mkdtemp(prefix="jude_whole_") + "/ds"
    jude._lance.write(t, whole, mode="create")
    shard_paths = []
    rows_per = math.ceil(n / S)
    for s in range(S):
        sub = t.slice(s * rows_per, rows_per)
        if sub.num_rows == 0:
            continue
        sp = tempfile.mkdtemp(prefix=f"jude_shard{s}_") + "/ds"
        jude._lance.write(sub, sp, mode="create")
        con = jude.connect()
        con.create_lance_vector_index(sp, "v", index_type="IVF_FLAT", metric="cosine",
                                      num_partitions=max(1, int(math.sqrt(sub.num_rows))))
        shard_paths.append(sp)

    # single-node ANN indexes on the whole corpus: IVF_FLAT and IVF_HNSW_SQ
    con1 = jude.connect()
    con1.execute("SET threads=1")
    con1.register("emb", t)
    whole_hnsw = tempfile.mkdtemp(prefix="jude_whole_hnsw_") + "/ds"
    jude._lance.write(t, whole_hnsw, mode="create")
    con_ann = jude.connect()
    con_ann.create_lance_vector_index(whole, "v", index_type="IVF_FLAT", metric="cosine",
                                      num_partitions=max(1, int(math.sqrt(n))))
    hnsw_ok = True
    try:
        jude.connect().create_lance_vector_index(
            whole_hnsw, "v", index_type="IVF_HNSW_SQ", metric="cosine",
            num_partitions=max(1, int(math.sqrt(n))))
    except Exception as e:  # noqa: BLE001
        hnsw_ok = False
        print(f"(IVF_HNSW_SQ unavailable: {e})")

    # ground truth for recall
    exact_ids0 = vector.knn(con1, "emb", "v", queries[0], k=k).column("id").to_pylist()

    print("-" * 78)
    print(f"  {'method':<42}{'ms/query':>10}{'QPS':>8}{'recall':>10}")
    print("  " + "-" * 68)

    # 1. single-node exact, 1 core
    lat, qps = _bench(lambda q: vector.knn(con1, "emb", "v", q, k=k), queries)
    print(f"  {'single-node EXACT (1 core)':<42}{lat:>9.1f}{qps:>8.0f}{'100%':>10}")

    # 2. single-node ANN (IVF_FLAT), reuse handle
    def sn_ann(q):
        return vector.knn_rerank(whole, "v", q, k=k, overfetch=5, nprobes=int(math.sqrt(n)), metric="cosine")
    lat, qps = _bench(sn_ann, queries)
    r = vector.recall_at_k(sn_ann(queries[0]).column("id").to_pylist(), exact_ids0, k)
    print(f"  {'single-node ANN IVF_FLAT (1 proc)':<42}{lat:>9.1f}{qps:>8.0f}{r:>10.0%}")

    # 2b. single-node ANN IVF_HNSW_SQ (HNSW graph + scalar quantization)
    if hnsw_ok:
        def sn_hnsw(q):
            return vector.knn_rerank(whole_hnsw, "v", q, k=k, overfetch=5, nprobes=int(math.sqrt(n)), metric="cosine")
        lat, qps = _bench(sn_hnsw, queries)
        r = vector.recall_at_k(sn_hnsw(queries[0]).column("id").to_pylist(), exact_ids0, k)
        print(f"  {'single-node ANN IVF_HNSW_SQ (1 proc)':<42}{lat:>9.1f}{qps:>8.0f}{r:>10.0%}")

    # 3. distributed resident exact, S workers
    lat, qps = _bench(lambda q: vector.distributed_knn_resident(shard_paths, "v", q, k=k, runner=runner), queries)
    rr = vector.distributed_knn_resident(shard_paths, "v", queries[0], k=k, runner=runner).column("id").to_pylist()
    r = vector.recall_at_k(rr, exact_ids0, k)
    print(f"  {f'distributed RESIDENT exact ({S} workers)':<42}{lat:>9.1f}{qps:>8.0f}{r:>10.0%}")

    # 4. distributed_ann_knn, S sharded indexes
    def dann(q):
        return vector.distributed_ann_knn(shard_paths, "v", q, k=k, overfetch=5,
                                          nprobes=int(math.sqrt(rows_per)), metric="cosine", runner=runner)
    lat, qps = _bench(dann, queries)
    r = vector.recall_at_k(dann(queries[0]).column("id").to_pylist(), exact_ids0, k)
    print(f"  {f'distributed_ann_knn ({S} shards)':<42}{lat:>9.1f}{qps:>8.0f}{r:>10.0%}")
    print("-" * 78)


if __name__ == "__main__":
    main()

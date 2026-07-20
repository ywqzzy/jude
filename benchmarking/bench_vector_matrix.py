#!/usr/bin/env python3
"""Comprehensive vector-search benchmark matrix — fixed N=1M.

Sweeps every dimension and reports build time, recall@k, latency p50/p95, QPS:
  A. single-node index types: exact, IVF_FLAT, IVF_PQ, IVF_HNSW_SQ (1 proc)
  B. workers sweep (1/2/4): distributed resident-exact + distributed_ann_knn
Everything at a FIXED 1M vectors so numbers are comparable across the matrix.

    python benchmarking/bench_vector_matrix.py --n 1000000 --dim 96 --k 100 --queries 30
"""

from __future__ import annotations

import argparse
import math
import statistics
import tempfile
import time

import numpy as np
import pyarrow as pa


def make(n, d, clusters=200, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    q = [(c[i % clusters] + 0.15 * rng.standard_normal(d)).astype("float32").tolist() for i in range(30)]
    return v, c, q


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0.0


def _measure(fn, queries):
    fn(queries[0])
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        lat.append((time.perf_counter() - t0) * 1000)
    return _pct(lat, 50), _pct(lat, 95), 1000 / statistics.mean(lat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=30)
    args = ap.parse_args()

    import jude
    from jude import vector
    from _bench_ray import connect_ray

    connect_ray(num_cpus=4)
    from jude.runners.ray import RayRunner

    n, d, k = args.n, args.dim, args.k
    v, centers, queries = make(n, d)
    queries = queries[: args.queries]
    t = pa.table({"id": list(range(n)), "v": pa.array(v.tolist(), type=pa.list_(pa.float32(), d))})
    raw_gb = n * d * 4 / 2**30
    print(f"\n{'='*72}\nVECTOR BENCHMARK MATRIX — N={n:,} dim={d} k={k} queries={len(queries)}")
    print(f"raw vectors: {raw_gb:.2f} GiB | data=clustered(200)\n{'='*72}")

    con1 = jude.connect()
    con1.execute("SET threads=1")
    con1.register("emb", t)
    exact_ids0 = vector.knn(con1, "emb", "v", queries[0], k=k).column("id").to_pylist()

    # ---- A. single-node index types (1 proc) ----
    print("\nA. single-node (1 core / 1 proc)")
    print(f"  {'method':<26}{'build s':>9}{'recall':>9}{'p50 ms':>9}{'p95 ms':>9}{'QPS':>8}")
    print("  " + "-" * 68)
    p50, p95, qps = _measure(lambda q: vector.knn(con1, "emb", "v", q, k=k), queries)
    print(f"  {'exact (1 core)':<26}{'-':>9}{'100%':>9}{p50:>9.1f}{p95:>9.1f}{qps:>8.0f}")

    nparts = max(1, int(math.sqrt(n)))
    for itype, sv in [("IVF_FLAT", None), ("IVF_PQ", max(1, d // 8)), ("IVF_HNSW_SQ", None)]:
        wp = tempfile.mkdtemp(prefix=f"jude_{itype}_") + "/ds"
        jude._lance.write(t, wp, mode="create")
        kw = {"index_type": itype, "metric": "cosine", "num_partitions": nparts}
        if sv:
            kw["num_sub_vectors"] = sv
        try:
            t0 = time.perf_counter()
            jude.connect().create_lance_vector_index(wp, "v", **kw)
            build = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  {itype:<26} (unavailable: {str(e)[:30]})")
            continue

        def fn(q, _wp=wp):
            return vector.knn_rerank(_wp, "v", q, k=k, overfetch=5, nprobes=nparts, metric="cosine")
        p50, p95, qps = _measure(fn, queries)
        r = vector.recall_at_k(fn(queries[0]).column("id").to_pylist(), exact_ids0, k)
        print(f"  {itype+' +rerank':<26}{build:>9.1f}{r:>9.0%}{p50:>9.1f}{p95:>9.1f}{qps:>8.0f}")

    # ---- B. workers sweep ----
    print("\nB. distributed workers sweep (data resident on workers)")
    print(f"  {'method':<34}{'recall':>9}{'p50 ms':>9}{'p95 ms':>9}{'QPS':>8}")
    print("  " + "-" * 68)
    for W in [1, 2, 4]:
        runner = RayRunner(num_workers=W)
        rows_per = math.ceil(n / W)
        shard_paths = []
        for s in range(W):
            sub = t.slice(s * rows_per, rows_per)
            if sub.num_rows == 0:
                continue
            sp = tempfile.mkdtemp(prefix=f"jude_w{W}s{s}_") + "/ds"
            jude._lance.write(sub, sp, mode="create")
            jude.connect().create_lance_vector_index(sp, "v", index_type="IVF_FLAT", metric="cosine",
                                                     num_partitions=max(1, int(math.sqrt(sub.num_rows))))
            shard_paths.append(sp)
        # resident exact
        fn_ex = lambda q: vector.distributed_knn_resident(shard_paths, "v", q, k=k, runner=runner)
        p50, p95, qps = _measure(fn_ex, queries)
        r = vector.recall_at_k(fn_ex(queries[0]).column("id").to_pylist(), exact_ids0, k)
        print(f"  {f'resident EXACT ({W} workers)':<34}{r:>9.0%}{p50:>9.1f}{p95:>9.1f}{qps:>8.0f}")
        # sharded ANN
        fn_ann = lambda q: vector.distributed_ann_knn(shard_paths, "v", q, k=k, overfetch=5,
                                                      nprobes=max(1, int(math.sqrt(rows_per))), metric="cosine", runner=runner)
        p50, p95, qps = _measure(fn_ann, queries)
        r = vector.recall_at_k(fn_ann(queries[0]).column("id").to_pylist(), exact_ids0, k)
        print(f"  {f'sharded ANN IVF_FLAT ({W} shards)':<34}{r:>9.0%}{p50:>9.1f}{p95:>9.1f}{qps:>8.0f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

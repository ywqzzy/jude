#!/usr/bin/env python3
"""WHERE THE INDEX WINS — nprobes sweep at a realistic embedding dim.

The earlier matrix made IVF look pointless because it set nprobes = num_partitions
(probe EVERY cell = no pruning = a full scan with index overhead on top), at a tiny
dim=96 where the exact scan is already cheap. Both were wrong. An IVF index wins by
probing FEW cells (nprobes << num_partitions), and its advantage grows with dim
(exact is O(N*d); IVF touches only nprobes/num_partitions of N).

This bench fixes both: dim=768 (real embedding size) and a proper nprobes sweep.
Over-fetch + exact re-rank keeps recall high while nprobes stays small.

  exact baseline (1 core) vs IVF_FLAT at nprobes in {1,4,8,16,32,64,128,256,1024}
  reports recall@k + QPS + speedup; marks the knee (recall>=95% at max speedup).
Also one IVF_PQ + one IVF_HNSW_SQ row for the algorithm comparison.

    python benchmarking/bench_vector_index_wins.py --n 1000000 --dim 768 --k 100
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
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
    q = (c[rng.integers(0, clusters, 64)] + 0.15 * rng.standard_normal((64, d))).astype("float32")
    return v, q


def to_table(v, ids):
    child = pa.array(v.reshape(-1), type=pa.float32())
    return pa.table({"id": pa.array(ids, type=pa.int64()),
                     "v": pa.FixedSizeListArray.from_arrays(child, v.shape[1])})


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def measure(fn, queries, warmup=3):
    for i in range(warmup):
        fn(queries[i % len(queries)])
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        lat.append((time.perf_counter() - t0) * 1000)
    return pct(lat, 50), pct(lat, 95), 1000 / statistics.mean(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--parts", type=int, default=0, help="num_partitions (0=sqrt(N))")
    args = ap.parse_args()

    import jude
    from jude import vector

    n, d, k = args.n, args.dim, args.k
    parts = args.parts or max(1, int(math.sqrt(n)))
    v, qm = make(n, d)
    queries = [q.tolist() for q in qm[: args.queries]]

    print("=" * 76)
    print(f"INDEX-WINS — N={n:,} dim={d} k={k} num_partitions={parts} queries={len(queries)}")
    print(f"raw vectors: {n*d*4/2**30:.2f} GiB")
    print("=" * 76)

    # exact baseline (1 core)
    con = jude.connect()
    con.execute("SET threads=1")
    con.register("emb", to_table(v, np.arange(n)))
    exact_ids = {}
    for i, q in enumerate(queries):
        exact_ids[i] = vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
    p50, p95, exact_qps = measure(lambda q: vector.knn(con, "emb", "v", q, k=k), queries)
    print(f"\nexact brute force (1 core): recall=100%  p50={p50:.1f}ms  QPS={exact_qps:.0f}")

    # build IVF_FLAT once
    wp = tempfile.mkdtemp(prefix="jude_ivfflat_") + "/ds"
    jude._lance.write(to_table(v, np.arange(n)), wp, mode="create")
    t0 = time.perf_counter()
    jude.connect().create_lance_vector_index(wp, "v", index_type="IVF_FLAT",
                                              metric="cosine", num_partitions=parts)
    build = time.perf_counter() - t0
    print(f"IVF_FLAT build: {build:.1f}s ({parts} partitions)\n")

    print(f"IVF_FLAT + over-fetch re-rank — nprobes sweep (overfetch=5)")
    print(f"  {'nprobes':>9}{'recall':>9}{'p50 ms':>9}{'p95 ms':>9}{'QPS':>8}{'vs exact':>10}")
    print("  " + "-" * 56)
    knee = None
    for nprobes in [1, 4, 8, 16, 32, 64, 128, 256, parts]:
        def fn(q, npr=nprobes):
            return vector.knn_rerank(wp, "v", q, k=k, overfetch=5, nprobes=npr, metric="cosine")
        recalls = []
        for i, q in enumerate(queries):
            got = fn(q).column("id").to_pylist()
            recalls.append(vector.recall_at_k(got, exact_ids[i], k))
        rec = sum(recalls) / len(recalls)
        p50, p95, qps = measure(fn, queries)
        tag = ""
        if rec >= 0.95 and knee is None:
            knee = nprobes
            tag = "  <- knee (>=95%)"
        print(f"  {nprobes:>9}{rec:>9.1%}{p50:>9.1f}{p95:>9.1f}{qps:>8.0f}{qps/exact_qps:>9.1f}x{tag}")

    # other index types at the knee nprobes for the algorithm comparison
    npr = knee or 32
    print(f"\nother index types (nprobes={npr}, overfetch=5):")
    print(f"  {'index':>14}{'build s':>9}{'recall':>9}{'p50 ms':>9}{'QPS':>8}{'vs exact':>10}")
    print("  " + "-" * 59)
    for itype, extra in [("IVF_PQ", {"num_sub_vectors": max(1, d // 8)}),
                         ("IVF_HNSW_SQ", {})]:
        p2 = tempfile.mkdtemp(prefix=f"jude_{itype}_") + "/ds"
        jude._lance.write(to_table(v, np.arange(n)), p2, mode="create")
        try:
            t0 = time.perf_counter()
            jude.connect().create_lance_vector_index(p2, "v", index_type=itype, metric="cosine",
                                                     num_partitions=parts, **extra)
            b = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  {itype:>14}  (unavailable: {str(e)[:30]})")
            continue

        def fn2(q, _p=p2):
            return vector.knn_rerank(_p, "v", q, k=k, overfetch=5, nprobes=npr, metric="cosine")
        recalls = []
        for i, q in enumerate(queries):
            recalls.append(vector.recall_at_k(fn2(q).column("id").to_pylist(), exact_ids[i], k))
        rec = sum(recalls) / len(recalls)
        p50, p95, qps = measure(fn2, queries)
        print(f"  {itype:>14}{b:>9.1f}{rec:>9.1%}{p50:>9.1f}{qps:>8.0f}{qps/exact_qps:>9.1f}x")
        shutil.rmtree(os.path.dirname(p2), ignore_errors=True)
    shutil.rmtree(os.path.dirname(wp), ignore_errors=True)
    print("=" * 76)


if __name__ == "__main__":
    main()

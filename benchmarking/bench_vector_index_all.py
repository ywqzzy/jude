#!/usr/bin/env python3
"""EXHAUSTIVE index benchmark — every supported Lance index type, every aspect.

Fixed N=1M, realistic embedding dim (default 768). For EACH index type jude
supports, reports the full picture:
  build time, on-disk index size, recall@k, latency p50/p95, QPS,
  and COLD (index not in memory) vs WARM (index cached in memory) QPS.

Index types probed (skipped with a note if the Lance build doesn't support them):
  IVF_FLAT, IVF_SQ, IVF_PQ, IVF_HNSW_FLAT, IVF_HNSW_SQ, IVF_HNSW_PQ
Plus: exact brute-force baseline, and an IVF_FLAT nprobes sweep.

The "cold vs warm" columns answer: can the index be cached in memory? WARM reuses
a dataset handle whose Lance index-cache holds the whole IVF resident; COLD drops
the handle each query so the index is re-read from disk. (jude._lance.set_index_cache_size)

    python benchmarking/bench_vector_index_all.py --n 1000000 --dim 768 --k 100
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import subprocess
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


def dir_mb(path):
    """Size of a Lance dataset's _indices dir in MB (the index footprint)."""
    idx = os.path.join(path, "_indices")
    if not os.path.isdir(idx):
        return 0.0
    try:
        out = subprocess.run(["du", "-sk", idx], capture_output=True, text=True, check=True)
        return int(out.stdout.split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--nprobes", type=int, default=16)
    ap.add_argument("--overfetch", type=int, default=5)
    args = ap.parse_args()

    import jude
    from jude import vector, _lance

    n, d, k = args.n, args.dim, args.k
    parts = max(1, int(math.sqrt(n)))
    nsv = max(1, d // 8)  # PQ sub-vectors
    v, qm = make(n, d)
    queries = [q.tolist() for q in qm[: args.queries]]

    print("=" * 92)
    print(f"EXHAUSTIVE INDEX BENCH — N={n:,} dim={d} k={k} nprobes={args.nprobes} "
          f"overfetch={args.overfetch} queries={len(queries)}")
    print(f"raw vectors: {n*d*4/2**30:.2f} GiB | num_partitions={parts} | pq_sub_vectors={nsv}")
    print("=" * 92)

    # exact ground truth + baseline
    con = jude.connect()
    con.execute("SET threads=1")
    con.register("emb", to_table(v, np.arange(n)))
    exact_ids = {i: vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
                 for i, q in enumerate(queries)}
    p50, p95, exact_qps = measure(lambda q: vector.knn(con, "emb", "v", q, k=k), queries)
    print(f"\nexact brute force (1 core): recall=100%  p50={p50:.1f}ms  p95={p95:.1f}ms  QPS={exact_qps:.1f}\n")

    index_types = [
        ("IVF_FLAT", {}),
        ("IVF_SQ", {}),
        ("IVF_PQ", {"num_sub_vectors": nsv}),
        ("IVF_HNSW_FLAT", {}),
        ("IVF_HNSW_SQ", {}),
        ("IVF_HNSW_PQ", {"num_sub_vectors": nsv}),
    ]

    hdr = (f"  {'index':<16}{'build s':>9}{'size MB':>9}{'recall':>9}{'p50 ms':>9}"
           f"{'p95 ms':>9}{'warm QPS':>10}{'cold QPS':>10}{'vs exact':>10}")
    print("ALL INDEX TYPES (nprobes={}, overfetch={}):".format(args.nprobes, args.overfetch))
    print(hdr)
    print("  " + "-" * 90)

    _lance.set_index_cache_size(100_000)  # pin the whole index in memory

    for itype, extra in index_types:
        p = tempfile.mkdtemp(prefix=f"jude_{itype}_") + "/ds"
        jude._lance.write(to_table(v, np.arange(n)), p, mode="create")
        try:
            t0 = time.perf_counter()
            jude.connect().create_lance_vector_index(p, "v", index_type=itype, metric="cosine",
                                                     num_partitions=parts, **extra)
            build = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  {itype:<16} unavailable: {str(e)[:60]}")
            shutil.rmtree(os.path.dirname(p), ignore_errors=True)
            continue
        size_mb = dir_mb(p)

        def warm_fn(q, _p=p):
            return vector.knn_rerank(_p, "v", q, k=k, overfetch=args.overfetch,
                                     nprobes=args.nprobes, metric="cosine")
        # recall
        recalls = [vector.recall_at_k(warm_fn(q).column("id").to_pylist(), exact_ids[i], k)
                   for i, q in enumerate(queries)]
        rec = sum(recalls) / len(recalls)
        # warm (index cached in memory)
        p50, p95, warm_qps = measure(warm_fn, queries)

        # cold (drop the handle each query -> index re-read from disk)
        def cold_fn(q, _p=p):
            jude._lance._DS_CACHE.pop(_p, None)
            return vector.knn_rerank(_p, "v", q, k=k, overfetch=args.overfetch,
                                     nprobes=args.nprobes, metric="cosine")
        _, _, cold_qps = measure(cold_fn, queries, warmup=1)

        print(f"  {itype:<16}{build:>9.1f}{size_mb:>9.1f}{rec:>9.1%}{p50:>9.1f}"
              f"{p95:>9.1f}{warm_qps:>10.1f}{cold_qps:>10.1f}{warm_qps/exact_qps:>9.1f}x")
        shutil.rmtree(os.path.dirname(p), ignore_errors=True)

    print("=" * 92)
    print("warm = index resident in memory (set_index_cache_size); cold = re-read per query.")


if __name__ == "__main__":
    main()

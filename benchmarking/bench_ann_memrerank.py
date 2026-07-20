#!/usr/bin/env python3
"""Diagnose + fix the ANN QPS bottleneck at realistic dim.

Plain knn_rerank fetches k*overfetch candidate VECTORS out of Lance per query
(Arrow materialization) — slow at dim=768. knn_ann_resident fetches candidate
IDs only and re-ranks against an in-RAM matrix. This proves which stage is the
bottleneck and how much the in-memory re-rank recovers.

Compares, at fixed N/dim, same recall target:
  exact (1 core)                         — baseline
  knn_rerank (Lance fetch + rerank)      — current path
  lance nearest, columns=[id] only       — IVF search cost WITHOUT vector fetch
  lance nearest, WITH vectors            — IVF search + vector fetch
  knn_ann_resident (IVF ids + RAM rerank)— the fix

    python benchmarking/bench_ann_memrerank.py --n 1000000 --dim 768 --k 100
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
    ap.add_argument("--nprobes", type=int, default=16)
    ap.add_argument("--overfetch", type=int, default=5)
    args = ap.parse_args()

    import jude
    from jude import vector, _lance

    n, d, k = args.n, args.dim, args.k
    parts = max(1, int(math.sqrt(n)))
    v, qm = make(n, d)
    queries = [q.tolist() for q in qm[: args.queries]]
    npr, ov = args.nprobes, args.overfetch

    print("=" * 80)
    print(f"ANN MEM-RERANK — N={n:,} dim={d} k={k} nprobes={npr} overfetch={ov} queries={len(queries)}")
    print("=" * 80)

    con = jude.connect()
    con.execute("SET threads=1")
    con.register("emb", to_table(v, np.arange(n)))
    exact_ids = {i: vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
                 for i, q in enumerate(queries)}
    p50, p95, exact_qps = measure(lambda q: vector.knn(con, "emb", "v", q, k=k), queries)
    print(f"\nexact (1 core):              recall=100%  p50={p50:6.1f}ms  QPS={exact_qps:6.1f}")

    wp = tempfile.mkdtemp(prefix="jude_memrr_") + "/ds"
    jude._lance.write(to_table(v, np.arange(n)), wp, mode="create")
    t0 = time.perf_counter()
    jude.connect().create_lance_vector_index(wp, "v", index_type="IVF_FLAT",
                                              metric="cosine", num_partitions=parts)
    print(f"IVF_FLAT build: {time.perf_counter()-t0:.1f}s\n")

    ds = _lance.dataset_cached(wp)

    def near_id(q):
        return ds.to_table(nearest={"column": "v", "q": [float(x) for x in q], "k": k * ov, "nprobes": npr},
                           columns=["id"])
    def near_vec(q):
        return ds.to_table(nearest={"column": "v", "q": [float(x) for x in q], "k": k * ov, "nprobes": npr})
    def rerank(q):
        return vector.knn_rerank(wp, "v", q, k=k, overfetch=ov, nprobes=npr, metric="cosine")
    def memrr(q):
        return vector.knn_ann_resident(wp, "v", q, k=k, overfetch=ov, nprobes=npr, metric="cosine")

    rows = []
    p50, p95, qps = measure(near_id, queries)
    rows.append(("lance nearest, id-only (IVF search)", "-", p50, p95, qps))
    p50, p95, qps = measure(near_vec, queries)
    rows.append(("lance nearest, WITH vectors (fetch)", "-", p50, p95, qps))
    rec = np.mean([vector.recall_at_k(rerank(queries[i]).column("id").to_pylist(), exact_ids[i], k)
                   for i in range(len(queries))])
    p50, p95, qps = measure(rerank, queries)
    rows.append(("knn_rerank (Lance fetch + rerank)", f"{rec:.1%}", p50, p95, qps))
    rec = np.mean([vector.recall_at_k(memrr(queries[i]).column("id").to_pylist(), exact_ids[i], k)
                   for i in range(len(queries))])
    p50, p95, qps = measure(memrr, queries)
    rows.append(("knn_ann_resident (IVF ids + RAM rerank)", f"{rec:.1%}", p50, p95, qps))

    print(f"  {'method':<42}{'recall':>8}{'p50 ms':>9}{'p95 ms':>9}{'QPS':>8}{'vs exact':>10}")
    print("  " + "-" * 84)
    for name, rec, p50, p95, qps in rows:
        print(f"  {name:<42}{rec:>8}{p50:>9.1f}{p95:>9.1f}{qps:>8.1f}{qps/exact_qps:>9.1f}x")
    print("=" * 80)
    shutil.rmtree(os.path.dirname(wp), ignore_errors=True)


if __name__ == "__main__":
    main()

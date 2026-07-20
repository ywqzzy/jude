#!/usr/bin/env python3
"""Scientific vector recall/latency/QPS benchmark for the billion-scale design.

Runs at the largest scale feasible on one machine and measures — over MANY
queries (not one) — recall@k against exact ground truth, plus latency p50/p95
and QPS, sweeping the knobs that govern the recall/latency tradeoff:
index type (IVF_FLAT vs IVF_PQ), nprobes, overfetch. Also times index build and
estimates memory, and extrapolates to 1B. This is the evidence behind
docs/billion_scale_vector_search.zh.md.

    python benchmarking/bench_billion_scale.py --n 500000 --dim 96 --k 100 --queries 50
"""

from __future__ import annotations

import argparse
import math
import statistics
import tempfile
import time

import numpy as np
import pyarrow as pa


def make_clustered(n, d, clusters, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    vecs = (centers[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    return vecs, centers


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))
    return xs[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--clusters", type=int, default=200)
    args = ap.parse_args()

    import jude
    from jude import vector

    n, d, k, Q = args.n, args.dim, args.k, args.queries
    print(f"\n=== Vector recall/latency benchmark ===")
    print(f"N={n:,}  dim={d}  top-k={k}  queries={Q}  data=clustered({args.clusters})")

    vecs, centers = make_clustered(n, d, args.clusters)
    # memory footprint of the raw vectors
    raw_gb = n * d * 4 / 2**30
    print(f"raw vectors: {raw_gb:.2f} GiB in memory")

    path = tempfile.mkdtemp(prefix="jude_bscale_") + "/ds"
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    jude._lance.write(t, path, mode="create")
    con = jude.connect()
    con.register("emb", t)

    # queries: perturbed cluster centers (realistic)
    rng = np.random.default_rng(12345)
    queries = [(centers[i % args.clusters] + 0.15 * rng.standard_normal(d)).astype("float32").tolist()
               for i in range(Q)]

    # ---- ground truth: exact top-k per query (+ exact latency) ----
    print("\ncomputing exact ground truth + exact latency ...")
    gt = []
    exact_lat = []
    for q in queries:
        t0 = time.perf_counter()
        ids = vector.knn(con, "emb", "v", q, k=k, metric="cosine").column("id").to_pylist()
        exact_lat.append((time.perf_counter() - t0) * 1000)
        gt.append(ids)
    print(f"EXACT: recall=100.0% (by definition)  "
          f"p50={_pct(exact_lat,50):.1f}ms p95={_pct(exact_lat,95):.1f}ms  "
          f"QPS={1000/statistics.mean(exact_lat):.0f}")

    nparts = max(1, int(math.sqrt(n)))

    def sweep(index_type, sub_vectors=None):
        kw = {"index_type": index_type, "metric": "cosine", "num_partitions": nparts}
        if sub_vectors:
            kw["num_sub_vectors"] = sub_vectors
        t0 = time.perf_counter()
        con.create_lance_vector_index(path, "v", **kw)
        build = time.perf_counter() - t0
        print(f"\n--- {index_type} (num_partitions={nparts}, build {build:.1f}s) ---")
        print(f"  {'nprobes':>8}{'overfetch':>10}{'recall@'+str(k):>12}{'p50 ms':>9}{'p95 ms':>9}{'QPS':>8}")
        for nprobes in [nparts // 16 or 1, nparts // 4 or 1, nparts]:
            for overfetch in [1, 5]:
                recs, lats = [], []
                for q, exact_ids in zip(queries, gt):
                    t0 = time.perf_counter()
                    ids = vector.knn_rerank(path, "v", q, k=k, overfetch=overfetch,
                                            nprobes=nprobes, metric="cosine").column("id").to_pylist()
                    lats.append((time.perf_counter() - t0) * 1000)
                    recs.append(vector.recall_at_k(ids, exact_ids, k))
                print(f"  {nprobes:>8}{overfetch:>10}{statistics.mean(recs):>11.1%}"
                      f"{_pct(lats,50):>9.1f}{_pct(lats,95):>9.1f}{1000/statistics.mean(lats):>8.0f}")

    sweep("IVF_FLAT")
    sweep("IVF_PQ", sub_vectors=max(1, d // 8))

    # ---- extrapolation to 1B ----
    print("\n=== extrapolation to 1B vectors (dim=768 float32) ===")
    per_vec = 768 * 4
    total_tb = 1e9 * per_vec / 2**40
    print(f"  raw storage @ dim768: {total_tb:.2f} TiB  -> must shard (single machine can't hold)")
    for shard_rows in (1_000_000, 5_000_000):
        S = math.ceil(1e9 / shard_rows)
        shard_gb = shard_rows * per_vec / 2**30
        print(f"  shard={shard_rows:,} rows -> S={S} shards, ~{shard_gb:.1f} GiB/shard "
              f"(fits one machine), IVF num_partitions~{int(math.sqrt(shard_rows))}")
    print("  query: fan out to S shards, each local ANN top-k' (k'~k/S x overfetch), driver merges.")
    print("  recall ~ per-shard recall (tune nprobes/overfetch); see IVF_FLAT rows above.")


if __name__ == "__main__":
    main()

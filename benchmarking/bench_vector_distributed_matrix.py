#!/usr/bin/env python3
"""Distributed vector-retrieval performance matrix — methods x index types x workers.

Attaches to the resident Ray (python -m jude.observe) if present. Measures every
DISTRIBUTED retrieval method jude has, and for the sharded/routed ANN methods,
sweeps the per-shard index type:

  A. resident EXACT       (distributed_knn_resident)        — 100% recall, per-query fan-out
  B. resident EXACT batch (distributed_knn_resident_batch)  — throughput, W scaling
  C. sharded ANN          (distributed_ann_knn)             — per-shard {IVF_FLAT,IVF_SQ,IVF_PQ,IVF_HNSW_SQ}
  D. cluster-routed ANN   (distributed_ann_knn_routed)      — route to n_shards_probe shards

Reports recall / p50 / QPS. Fixed N, sharded across W workers.

    python benchmarking/bench_vector_distributed_matrix.py --n 1000000 --dim 768 --workers 4
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


def make(n, d, clusters, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.12 * rng.standard_normal((n, d))).astype("float32")
    q = (c[rng.integers(0, clusters, 64)] + 0.12 * rng.standard_normal((64, d))).astype("float32")
    return v, lab, q


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
    return pct(lat, 50), 1000 / statistics.mean(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--nprobes", type=int, default=16)
    args = ap.parse_args()

    import jude
    from jude import vector
    from _bench_ray import connect_ray

    connect_ray(num_cpus=max(4, args.workers))
    from jude.runners.ray import RayRunner

    n, d, k, W = args.n, args.dim, args.k, args.workers
    clusters = W * 8  # a few clusters per shard
    v, lab, qm = make(n, d, clusters)
    queries = [q.tolist() for q in qm[: args.queries]]
    runner = RayRunner(num_workers=W)

    print("=" * 84)
    print(f"DISTRIBUTED VECTOR MATRIX — N={n:,} dim={d} k={k} workers={W} nprobes={args.nprobes}")
    print("=" * 84)

    # exact ground truth (single-node, 1 core)
    con = jude.connect(); con.execute("SET threads=1"); con.register("emb", to_table(v, np.arange(n)))
    exact_ids = {i: vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
                 for i, q in enumerate(queries)}

    # ---- shard the corpus BY CLUSTER (so routing has meaning) ----
    # assign each cluster to a shard; a shard = union of its clusters' rows.
    cl_to_shard = {c: c % W for c in range(clusters)}
    shard_rows = [[] for _ in range(W)]
    for i in range(n):
        shard_rows[cl_to_shard[int(lab[i])]].append(i)
    shard_paths, shard_cents = [], []
    tmp = []
    for s in range(W):
        idx = np.array(shard_rows[s], dtype=np.int64)
        p = tempfile.mkdtemp(prefix=f"jude_dm_s{s}_") + "/ds"
        tmp.append(os.path.dirname(p))
        jude._lance.write(to_table(v[idx], idx), p, mode="create")
        shard_paths.append(p)
        shard_cents.append(v[idx].mean(axis=0).tolist())

    def recall_of(fn):
        return float(np.mean([vector.recall_at_k(fn(queries[i]).column("id").to_pylist(),
                                                 exact_ids[i], k) for i in range(len(queries))]))

    print(f"\n  {'method':<38}{'recall':>9}{'p50 ms':>9}{'QPS':>8}")
    print("  " + "-" * 64)

    # A. resident exact (per-query fan-out)
    fn = lambda q: vector.distributed_knn_resident(shard_paths, "v", q, k=k, runner=runner)
    rec = recall_of(fn); p50, qps = measure(fn, queries)
    print(f"  {'resident EXACT (fan-out)':<38}{rec:>9.1%}{p50:>9.1f}{qps:>8.1f}")

    # B. resident exact BATCH (throughput)
    vector.distributed_knn_resident_batch(shard_paths, "v", queries, k=k, runner=runner)
    t0 = time.perf_counter()
    vector.distributed_knn_resident_batch(shard_paths, "v", queries, k=k, runner=runner)
    dt = time.perf_counter() - t0
    print(f"  {'resident EXACT batch (throughput)':<38}{'100%':>9}{'-':>9}{len(queries)/dt:>8.1f}")

    # build per-shard indexes and sweep index type for sharded + routed ANN
    for itype, extra in [("IVF_FLAT", {}), ("IVF_SQ", {}), ("IVF_PQ", {"num_sub_vectors": max(1, d // 8)}),
                         ("IVF_HNSW_SQ", {})]:
        try:
            for p in shard_paths:
                rows = jude._lance.dataset_cached(p).count_rows()
                jude.connect().create_lance_vector_index(p, "v", index_type=itype, metric="cosine",
                                                         num_partitions=max(1, int(math.sqrt(rows))),
                                                         replace=True, **extra)
            jude._lance._DS_CACHE.clear()
        except Exception as e:  # noqa: BLE001
            print(f"  sharded ANN [{itype}] index build failed: {str(e)[:40]}")
            continue

        fn = lambda q, npr=args.nprobes: vector.distributed_ann_knn(
            shard_paths, "v", q, k=k, overfetch=5, nprobes=npr, metric="cosine", runner=runner)
        rec = recall_of(fn); p50, qps = measure(fn, queries)
        print(f"  {'sharded ANN all-shards ['+itype+']':<38}{rec:>9.1%}{p50:>9.1f}{qps:>8.1f}")

        fn = lambda q, npr=args.nprobes: vector.distributed_ann_knn_routed(
            shard_paths, shard_cents, "v", q, k=k, n_shards_probe=max(1, W // 2),
            overfetch=5, nprobes=npr, metric="cosine", runner=runner)
        rec = recall_of(fn); p50, qps = measure(fn, queries)
        print(f"  {'routed ANN (probe '+str(max(1,W//2))+'/'+str(W)+') ['+itype+']':<38}{rec:>9.1%}{p50:>9.1f}{qps:>8.1f}")

    print("=" * 84)
    for t in tmp:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()

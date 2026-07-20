#!/usr/bin/env python3
"""Where distributed sharded ANN WINS over distributed resident-exact.

Resident-exact is a zero-overhead in-RAM BLAS matmul; ANN saves compute (scans
few vectors) but pays fixed per-query overhead (Lance scanner + IVF traversal +
candidate fetch + rerank). So ANN only wins once the PER-SHARD brute-force scan
is expensive — i.e. large per-shard N and/or high dim. This bench pushes
per-shard N up (few workers, large N, high dim) to cross that point, and reports
recall + p50 + QPS for both, single-query.

    python benchmarking/bench_vector_sharded_win.py --n 1000000 --dim 768 --workers 2
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
    q = (c[rng.integers(0, clusters, 40)] + 0.15 * rng.standard_normal((40, d))).astype("float32")
    return v, q


def to_table(v, ids):
    ch = pa.array(v.reshape(-1), type=pa.float32())
    return pa.table({"id": pa.array(ids, type=pa.int64()),
                     "v": pa.FixedSizeListArray.from_arrays(ch, v.shape[1])})


def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def measure(fn, qs, warm=3):
    for i in range(warm):
        fn(qs[i % len(qs)])
    lat = []
    for q in qs:
        a = time.perf_counter(); fn(q); lat.append((time.perf_counter() - a) * 1000)
    return pct(lat, 50), 1000 / statistics.mean(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--nprobes", type=int, default=24)
    args = ap.parse_args()

    import jude
    from jude import vector
    from _bench_ray import connect_ray

    connect_ray(num_cpus=max(4, args.workers))
    from jude.runners.ray import RayRunner

    n, d, W, k = args.n, args.dim, args.workers, args.k
    per = n // W
    v, qm = make(n, d)
    qs = [q.tolist() for q in qm[: args.queries]]

    print("=" * 78)
    print(f"SHARDED ANN WIN REGIME — N={n:,} dim={d} W={W} (per-shard={per:,}) k={k} nprobes={args.nprobes}")
    print("=" * 78)

    # exact ground truth (single-node, 1 core) for recall
    con = jude.connect(); con.execute("SET threads=1"); con.register("emb", to_table(v, np.arange(n)))
    exact = {i: vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist() for i, q in enumerate(qs)}

    rows = math.ceil(n / W); paths = []; tmp = []
    t0 = time.perf_counter()
    for s in range(W):
        lo, hi = s * rows, min((s + 1) * rows, n)
        p = tempfile.mkdtemp(prefix=f"jude_sw{s}_") + "/ds"; tmp.append(os.path.dirname(p))
        jude._lance.write(to_table(v[lo:hi], np.arange(lo, hi)), p, mode="create")
        jude.connect().create_lance_vector_index(p, "v", index_type="IVF_SQ", metric="cosine",
                                                  num_partitions=max(1, int(math.sqrt(hi - lo))))
        paths.append(p)
    print(f"built {W} shards + IVF index in {time.perf_counter()-t0:.0f}s\n")
    r = RayRunner(num_workers=W)

    def rec(fn):
        return float(np.mean([vector.recall_at_k(fn(qs[i]).column("id").to_pylist(), exact[i], k)
                              for i in range(len(qs))]))

    print(f"  {'method':<34}{'recall':>9}{'p50 ms':>10}{'QPS':>8}")
    print("  " + "-" * 62)
    fx = lambda q: vector.distributed_knn_resident(paths, "v", q, k=k, runner=r)
    rr = rec(fx); p50, qps = measure(fx, qs)
    print(f"  {'resident EXACT (brute, in-RAM)':<34}{rr:>9.1%}{p50:>10.1f}{qps:>8.1f}")
    fa = lambda q: vector.distributed_ann_knn(paths, "v", q, k=k, overfetch=5, nprobes=args.nprobes, runner=r)
    ra = rec(fa); p50, qps = measure(fa, qs)
    print(f"  {'sharded ANN (IVF per shard)':<34}{ra:>9.1%}{p50:>10.1f}{qps:>8.1f}")
    print("=" * 78)
    for t in tmp:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()

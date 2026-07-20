#!/usr/bin/env python3
"""WHY distributed vector search doesn't scale linearly — a scientific decomposition.

A single resident-KNN query must touch EVERY shard (each holds 1/W of the
vectors; the global top-k needs all of them). So its latency is

    dispatch(W tasks) + max_i(rpc + scan N/W) + gather(W) + merge(W*k)
        ^ grows w/ W        ^ shrinks w/ W        ^ grow w/ W

The per-shard scan (the only parallel part) shrinks with W, but the driver-side
coordination (submit W Ray tasks, ray.get W results, merge W top-k lists) is
serial and GROWS with W. When per-shard work is small, coordination dominates
and adding workers barely helps — Amdahl's law. This bench measures each piece.

Sections:
  1. COORDINATION FLOOR: single-process numpy scan (no Ray) vs W=1 vs W=4 — how
     much of a distributed query is compute vs Ray coordination.
  2. STRONG scaling: fixed N=1M, W in {1,2,4,8}, at dim=96 AND dim=768 — shows
     latency flattens (Amdahl), and that bigger per-shard work (768-d) scales
     better because compute out-weighs coordination.
  3. WEAK scaling: fixed rows-per-worker, N grows with W — isolates parallel
     efficiency (ideal = flat latency as the cluster and data grow together).
  4. THROUGHPUT: many queries in flight (batched) — the axis that DOES scale
     ~linearly, because coordination is amortized over the batch.

    python benchmarking/bench_vector_scaling.py
"""

from __future__ import annotations

import math
import shutil
import statistics
import tempfile
import time

import numpy as np
import pyarrow as pa


def make_matrix(n: int, d: int, clusters: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
    q = (c[rng.integers(0, clusters, 64)] + 0.15 * rng.standard_normal((64, d))).astype("float32")
    return v, q


def to_table(v: np.ndarray, ids: np.ndarray) -> pa.Table:
    """Build id + FixedSizeList<float32,d> WITHOUT tolist() (which would
    materialize N*d Python floats — fatal at d=768)."""
    n, d = v.shape
    child = pa.array(v.reshape(-1), type=pa.float32())
    vecs = pa.FixedSizeListArray.from_arrays(child, d)
    return pa.table({"id": pa.array(ids, type=pa.int64()), "v": vecs})


def write_shards(v: np.ndarray, w: int, tag: str):
    import jude

    n = v.shape[0]
    rows = math.ceil(n / w)
    paths = []
    for s in range(w):
        lo, hi = s * rows, min((s + 1) * rows, n)
        if lo >= hi:
            break
        p = tempfile.mkdtemp(prefix=f"jude_{tag}_w{w}s{s}_") + "/ds"
        jude._lance.write(to_table(v[lo:hi], np.arange(lo, hi)), p, mode="create")
        paths.append(p)
    return paths


def cleanup(paths):
    """Remove shard temp dirs — these are multi-GB at realistic dims and will
    fill the disk across a sweep if left behind."""
    import os

    for p in paths:
        shutil.rmtree(os.path.dirname(p), ignore_errors=True)


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
    mean = statistics.mean(lat)
    return pct(lat, 50), pct(lat, 95), 1000 / mean


def main():
    import jude
    from jude import vector
    from _bench_ray import connect_ray

    connect_ray(num_cpus=8)
    from jude.runners.ray import RayRunner

    N, K = 1_000_000, 100
    NQ = 50

    print("=" * 78)
    print(f"WHY-NO-LINEAR-SCALING — N={N:,} k={K} queries={NQ}")
    print("=" * 78)

    # ---- 1. coordination floor (dim=96) --------------------------------------
    d = 96
    v96, q96 = make_matrix(N, d)
    queries = [q.tolist() for q in q96[:NQ]]
    # pure single-process numpy scan — the compute floor, no Ray at all
    norms = np.linalg.norm(v96, axis=1)
    norms[norms == 0] = 1.0

    def numpy_scan(q):
        qv = np.asarray(q, dtype="float32")
        dist = 1.0 - (v96 @ qv) / (norms * (np.linalg.norm(qv) or 1.0))
        part = np.argpartition(dist, K - 1)[:K]
        return part[np.argsort(dist[part])]

    floor_p50, _, floor_qps = measure(numpy_scan, queries)

    print(f"\n1. COORDINATION FLOOR (dim={d}, N={N:,})")
    print(f"   {'config':<40}{'p50 ms':>10}{'QPS':>8}{'note':>18}")
    print("   " + "-" * 74)
    print(f"   {'pure numpy scan (no Ray, 1 proc)':<40}{floor_p50:>10.2f}{floor_qps:>8.0f}"
          f"{'compute floor':>18}")
    for w in (1, 4):
        paths = write_shards(v96, w, "coord")
        runner = RayRunner(num_workers=w)
        fn = lambda q, p=paths, r=runner: vector.distributed_knn_resident(p, "v", q, k=K, runner=r)
        p50, _, qps = measure(fn, queries)
        overhead = p50 - floor_p50 / w  # measured minus ideal (floor/w)
        print(f"   {f'resident W={w} (per-shard scan={floor_p50/w:.2f}ms ideal)':<40}"
              f"{p50:>10.2f}{qps:>8.0f}{f'+{overhead:.2f}ms coord':>18}")
        cleanup(paths)
    print("   -> the gap above the ideal floor/W is Ray dispatch+gather+merge; it")
    print("      does NOT shrink with W, so single-query latency can't go linear.")

    # ---- 2. strong scaling at two dims ---------------------------------------
    print(f"\n2. STRONG scaling (fixed N={N:,}, vary W) — latency p50 / QPS / speedup")
    print("   model: latency(W) ~= coord_floor + compute/W ; bigger dim -> compute")
    print("   dominates the fixed coord floor -> scaling gets closer to linear")
    for dlabel in ("dim=96", "dim=768"):
        if dlabel == "dim=96":
            v, qm = v96, q96
        else:
            v, qm = make_matrix(N, 768, seed=1)
        qs = [q.tolist() for q in qm[:NQ]]
        print(f"\n   [{dlabel}]  {'W':>3}{'p50 ms':>10}{'QPS':>8}{'speedup':>10}")
        print("   " + "-" * 40)
        base = None
        for w in (1, 2, 4, 8):
            paths = write_shards(v, w, f"strong{dlabel}")
            runner = RayRunner(num_workers=w)
            fn = lambda q, p=paths, r=runner: vector.distributed_knn_resident(p, "v", q, k=K, runner=r)
            p50, _, qps = measure(fn, qs)
            base = base or qps
            print(f"        {w:>3}{p50:>10.2f}{qps:>8.0f}{qps/base:>9.2f}x")
            cleanup(paths)
        if dlabel == "dim=768":
            del v, qm

    # ---- 3. weak scaling (rows/worker fixed) ---------------------------------
    print(f"\n3. WEAK scaling (dim=96, ~250k rows/worker fixed, N grows with W)")
    print(f"   ideal = FLAT latency as data+workers grow together")
    print(f"   {'W':>3}{'N':>12}{'p50 ms':>10}{'QPS':>8}")
    print("   " + "-" * 33)
    per = 250_000
    for w in (1, 2, 4, 8):
        n = per * w
        vv, qq = make_matrix(n, 96, seed=2)
        qs = [q.tolist() for q in qq[:NQ]]
        paths = write_shards(vv, w, "weak")
        runner = RayRunner(num_workers=w)
        fn = lambda q, p=paths, r=runner: vector.distributed_knn_resident(p, "v", q, k=K, runner=r)
        p50, _, qps = measure(fn, qs)
        print(f"   {w:>3}{n:>12,}{p50:>10.2f}{qps:>8.0f}")
        cleanup(paths)
        del vv, qq

    # ---- 4. throughput (batched, amortized coordination) ---------------------
    print(f"\n4. THROUGHPUT (dim=96, N={N:,}, batch of {NQ} queries in ONE RPC/worker)")
    print(f"   this is the axis that scales ~linearly (coordination amortized)")
    print(f"   {'W':>3}{'batch ms':>10}{'agg QPS':>10}{'speedup':>10}")
    print("   " + "-" * 33)
    qs_all = [q.tolist() for q in q96[:NQ]]
    base = None
    for w in (1, 2, 4, 8):
        paths = write_shards(v96, w, "thru")
        runner = RayRunner(num_workers=w)
        # warm caches
        vector.distributed_knn_resident_batch(paths, "v", qs_all, k=K, runner=runner)
        t0 = time.perf_counter()
        vector.distributed_knn_resident_batch(paths, "v", qs_all, k=K, runner=runner)
        dt = (time.perf_counter() - t0) * 1000
        qps = len(qs_all) / (dt / 1000)
        base = base or qps
        print(f"   {w:>3}{dt:>10.1f}{qps:>10.0f}{qps/base:>9.2f}x")
        cleanup(paths)
    print("=" * 78)


if __name__ == "__main__":
    main()

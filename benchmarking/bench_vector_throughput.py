#!/usr/bin/env python3
"""Distributed vector THROUGHPUT scaling — the axis where distributed wins.

A single query fanned across shards is Amdahl-bound (fixed dispatch+gather+merge
> the parallelized per-shard compute at small N). But THROUGHPUT — many queries
in flight — scales with workers, because coordination is amortized across the
batch and every worker stays busy. This bench shows aggregate QPS vs #workers.

For each W in {1,2,4,8}: shard N across W workers, then push a big query batch and
measure aggregate QPS (queries / wall-time). Two engines:
  - resident EXACT batch  (100% recall)
  - sharded ANN batch     (per-shard IVF, driver merges) — issued concurrently

    python benchmarking/bench_vector_throughput.py --n 1000000 --dim 768 --batch 200
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa


def make(n, d, clusters=200, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d)).astype("float32")
    lab = rng.integers(0, clusters, n)
    v = (c[lab] + 0.12 * rng.standard_normal((n, d))).astype("float32")
    q = (c[rng.integers(0, clusters, 512)] + 0.12 * rng.standard_normal((512, d))).astype("float32")
    return v, q


def to_table(v, ids):
    child = pa.array(v.reshape(-1), type=pa.float32())
    return pa.table({"id": pa.array(ids, type=pa.int64()),
                     "v": pa.FixedSizeListArray.from_arrays(child, v.shape[1])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--nprobes", type=int, default=16)
    args = ap.parse_args()

    import jude
    from jude import vector
    from _bench_ray import connect_ray

    connect_ray(num_cpus=8)
    from jude.runners.ray import RayRunner
    from jude.runners import _ray_shim as shim

    n, d, k = args.n, args.dim, args.k
    v, qm = make(n, d)
    batch = [q.tolist() for q in qm[: args.batch]]

    print("=" * 76)
    print(f"DISTRIBUTED THROUGHPUT — N={n:,} dim={d} k={k} batch={len(batch)}")
    print("single-node knn_ann_resident is the latency champ; this shows the")
    print("THROUGHPUT axis where MORE WORKERS win (aggregate QPS).")
    print("=" * 76)
    print(f"\n  {'engine':<26}{'W':>3}{'batch ms':>10}{'agg QPS':>10}{'speedup':>10}")
    print("  " + "-" * 58)

    for engine in ("resident EXACT batch", "sharded ANN (concurrent)"):
        base = None
        for W in (1, 2, 4, 8):
            rows = math.ceil(n / W)
            paths, tmp = [], []
            for s in range(W):
                lo, hi = s * rows, min((s + 1) * rows, n)
                if lo >= hi:
                    break
                p = tempfile.mkdtemp(prefix=f"jude_tp_{engine[:3]}{W}s{s}_") + "/ds"
                tmp.append(os.path.dirname(p))
                jude._lance.write(to_table(v[lo:hi], np.arange(lo, hi)), p, mode="create")
                if engine.startswith("sharded"):
                    jude.connect().create_lance_vector_index(
                        p, "v", index_type="IVF_FLAT", metric="cosine",
                        num_partitions=max(1, int(math.sqrt(hi - lo))))
                paths.append(p)
            runner = RayRunner(num_workers=W)

            if engine.startswith("resident"):
                vector.distributed_knn_resident_batch(paths, "v", batch, k=k, runner=runner)  # warm
                t0 = time.perf_counter()
                vector.distributed_knn_resident_batch(paths, "v", batch, k=k, runner=runner)
                dt = time.perf_counter() - t0
            else:
                # concurrent sharded-ANN: issue ALL queries' shard tasks in flight,
                # then gather — keeps every worker busy (throughput, not per-query latency).
                workers = runner._ensure_workers()
                def submit_all():
                    refs = []
                    for qi, q in enumerate(batch):
                        for i, p in enumerate(paths):
                            refs.append(workers[runner.mgr.worker_for(qi * len(paths) + i)]
                                        .vector_knn_shard.remote(p, "v", q, k, 5, args.nprobes, "cosine", None))
                    return shim.get(refs)
                submit_all()  # warm
                t0 = time.perf_counter()
                submit_all()
                dt = time.perf_counter() - t0
            qps = len(batch) / dt
            base = base or qps
            print(f"  {engine:<26}{W:>3}{dt*1000:>10.1f}{qps:>10.0f}{qps/base:>9.2f}x")
            for t in tmp:
                shutil.rmtree(t, ignore_errors=True)
        print()
    print("=" * 76)


if __name__ == "__main__":
    main()

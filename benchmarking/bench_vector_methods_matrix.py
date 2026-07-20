#!/usr/bin/env python3
"""Retrieval-method x index-type matrix. For every Lance index type, run BOTH
two-stage ANN paths and report recall + QPS:
  - knn_rerank        (Lance index -> fetch candidate vectors -> exact rerank)
  - knn_ann_resident  (Lance index -> candidate IDs -> in-RAM exact rerank)

Proves knn_ann_resident works with ALL index types, and shows the recall/QPS of
each index under the same exact-rerank. One dataset, index rebuilt in place per
type (replace=True) to save disk. Exact brute force is the recall ground truth.

    python benchmarking/bench_vector_methods_matrix.py --n 1000000 --dim 768 --k 100
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
    return pct(lat, 50), 1000 / statistics.mean(lat)


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
    nsv = max(1, d // 8)
    npr, ov = args.nprobes, args.overfetch
    v, qm = make(n, d)
    queries = [q.tolist() for q in qm[: args.queries]]

    print("=" * 88)
    print(f"METHOD x INDEX MATRIX — N={n:,} dim={d} k={k} nprobes={npr} overfetch={ov} queries={len(queries)}")
    print("=" * 88)

    con = jude.connect()
    con.execute("SET threads=1")
    con.register("emb", to_table(v, np.arange(n)))
    exact_ids = {i: vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
                 for i, q in enumerate(queries)}
    _, exact_qps = measure(lambda q: vector.knn(con, "emb", "v", q, k=k), queries)
    print(f"\nexact brute force (1 core): recall=100%  QPS={exact_qps:.1f}\n")

    wp = tempfile.mkdtemp(prefix="jude_mx_") + "/ds"
    jude._lance.write(to_table(v, np.arange(n)), wp, mode="create")
    _lance.set_index_cache_size(100_000)
    # prime the resident matrix once (shared across index types)
    vector._resident_vectors(wp, "v")

    index_types = [
        ("IVF_FLAT", {}), ("IVF_SQ", {}), ("IVF_PQ", {"num_sub_vectors": nsv}),
        ("IVF_HNSW_FLAT", {}), ("IVF_HNSW_SQ", {}), ("IVF_HNSW_PQ", {"num_sub_vectors": nsv}),
    ]

    print(f"  {'index':<16}{'build s':>9}"
          f"{'  | knn_rerank        ':<24}{'| knn_ann_resident':<22}")
    print(f"  {'':<16}{'':>9}{'recall':>9}{'QPS':>8}{'   ':>3}{'recall':>9}{'QPS':>8}")
    print("  " + "-" * 82)
    for itype, extra in index_types:
        try:
            t0 = time.perf_counter()
            jude.connect().create_lance_vector_index(wp, "v", index_type=itype, metric="cosine",
                                                     num_partitions=parts, replace=True, **extra)
            build = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  {itype:<16} unavailable: {str(e)[:50]}")
            continue
        _lance._DS_CACHE.pop(wp, None)  # pick up the rebuilt index

        def rr(q):
            return vector.knn_rerank(wp, "v", q, k=k, overfetch=ov, nprobes=npr, metric="cosine")
        def mr(q):
            return vector.knn_ann_resident(wp, "v", q, k=k, overfetch=ov, nprobes=npr, metric="cosine")
        rr_rec = np.mean([vector.recall_at_k(rr(queries[i]).column("id").to_pylist(), exact_ids[i], k)
                          for i in range(len(queries))])
        _, rr_qps = measure(rr, queries)
        mr_rec = np.mean([vector.recall_at_k(mr(queries[i]).column("id").to_pylist(), exact_ids[i], k)
                          for i in range(len(queries))])
        _, mr_qps = measure(mr, queries)
        print(f"  {itype:<16}{build:>9.1f}{rr_rec:>9.1%}{rr_qps:>8.1f}{'   ':>3}{mr_rec:>9.1%}{mr_qps:>8.1f}")
    print("=" * 88)
    shutil.rmtree(os.path.dirname(wp), ignore_errors=True)


if __name__ == "__main__":
    main()

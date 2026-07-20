#!/usr/bin/env python3
"""Vector recall benchmark — top-k retrieval recall vs exact ground truth.

Answers "top-100k with recall as high as possible": measures recall@k for
exact brute force (100%) and ANN configs, so you can pick a method that meets a
recall target. Key findings (see docs/vector_high_recall.zh.md):

- EXACT brute force is 100% recall and, at ≤ a few million vectors, is FAST
  (~0.01s for 100k×128) — for large k (e.g. 100k) it is usually both the
  highest-recall and the most practical choice.
- ANN needs the right index + tuning: IVF_FLAT (no compression) + exact re-rank
  reaches ~100% recall on realistic (clustered) embeddings; IVF_PQ compresses
  vectors and caps recall. `nprobes` is the recall knob (scan more cells).

    python benchmarking/bench_vector_recall.py --n 100000 --dim 128 --k 1000
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time

import numpy as np
import pyarrow as pa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--clusters", type=int, default=50, help="0 = pure noise (ANN worst case)")
    ap.add_argument("--index", default="IVF_FLAT", choices=["IVF_FLAT", "IVF_PQ"])
    args = ap.parse_args()

    import jude
    from jude import vector

    rng = np.random.default_rng(0)
    n, d, k = args.n, args.dim, args.k
    if args.clusters > 0:
        centers = rng.standard_normal((args.clusters, d)).astype("float32")
        lab = rng.integers(0, args.clusters, n)
        vecs = (centers[lab] + 0.15 * rng.standard_normal((n, d))).astype("float32")
        q = (centers[3 % args.clusters] + 0.15 * rng.standard_normal(d)).astype("float32").tolist()
    else:
        vecs = rng.standard_normal((n, d)).astype("float32")
        q = rng.standard_normal(d).astype("float32").tolist()

    path = tempfile.mkdtemp(prefix="jude_recall_bench_") + "/ds"
    t = pa.table({"id": list(range(n)), "v": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d))})
    jude._lance.write(t, path, mode="create")
    con = jude.connect()
    con.register("emb", t)

    exact = vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
    nparts = max(1, int(math.sqrt(n)))
    kw = {"num_partitions": nparts}
    if args.index == "IVF_PQ":
        kw["num_sub_vectors"] = 16
    t0 = time.perf_counter()
    con.create_lance_vector_index(path, "v", index_type=args.index, metric="cosine", **kw)
    build = time.perf_counter() - t0

    print(f"\nvector recall — N={n:,} dim={d} top-k={k} "
          f"({'clustered' if args.clusters else 'noise'}), index={args.index} "
          f"num_partitions={nparts} (build {build:.2f}s)")
    print("-" * 62)
    print(f"{'method':<40}{'recall@'+str(k):>12}{'time':>10}")
    print("-" * 62)

    t0 = time.perf_counter()
    ids = vector.knn(con, "emb", "v", q, k=k).column("id").to_pylist()
    print(f"{'EXACT brute force':<40}{vector.recall_at_k(ids, exact, k):>11.1%}{time.perf_counter()-t0:>9.3f}s")

    for npr in [20, 50, 100, 200, nparts]:
        t0 = time.perf_counter()
        ids = vector.knn_rerank(path, "v", q, k=k, overfetch=5, nprobes=npr).column("id").to_pylist()
        print(f"{f'{args.index} rerank of=5 nprobes={npr}':<40}"
              f"{vector.recall_at_k(ids, exact, k):>11.1%}{time.perf_counter()-t0:>9.3f}s")
    print("-" * 62)
    print("recommendation: for k this large on ≤millions of vectors, EXACT is 100% and fastest;")
    print("use IVF_FLAT+rerank (not IVF_PQ) when the dataset is too big to brute-force.")


if __name__ == "__main__":
    main()

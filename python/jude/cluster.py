"""jude.cluster — distributed k-means (map-reduce Lloyd) + scalable semantic dedup.

Distributed k-means: the per-point nearest-centroid scan + per-cluster
accumulation runs in Rust on each shard (map); the driver merges the partial
sums into new centroids and iterates (reduce). This is the classic scalable
k-means, and it unlocks **scalable semantic dedup**: cluster the embeddings
first, then dedup *within each cluster*, so we never do a global O(n^2) cosine
comparison.

    from jude import cluster
    centroids, labels = cluster.kmeans(table, column="embedding", k=256)
    deduped = cluster.semantic_dedup_clustered(table, column="embedding",
                                               n_clusters=256, threshold=0.9)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from .jude import _curate

__all__ = ["kmeans", "kmeans_distributed", "semantic_dedup_clustered"]


def _as_matrix(table: pa.Table, column: str) -> np.ndarray:
    raw = table.column(column).to_pylist()
    return np.asarray(raw, dtype="float32")


def _init_centroids(mat: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(mat.shape[0], size=min(k, mat.shape[0]), replace=False)
    return mat[idx].copy()


def kmeans(
    table: Any,
    column: str = "embedding",
    *,
    k: int = 256,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-node k-means (Rust hot loop). Returns (centroids[k,dim],
    labels[n]). Lloyd's algorithm: assign -> update -> repeat until the inertia
    stops improving (relative ``tol``) or ``max_iter``."""
    mat = _as_matrix(table if isinstance(table, pa.Table) else table.to_arrow(), column)
    n, dim = mat.shape
    kk = min(k, n)
    centroids = _init_centroids(mat, kk, seed)
    flat = mat.reshape(-1).tolist()
    last = float("inf")
    for _ in range(max_iter):
        sums, counts, inertia = _curate.kmeans_assign_accumulate(
            flat, n, dim, [c.tolist() for c in centroids]
        )
        for c in range(len(centroids)):
            if counts[c] > 0:
                centroids[c] = np.asarray(sums[c], dtype="float64").reshape(-1) / counts[c]
        centroids = centroids.astype("float32")
        if last - inertia <= tol * max(1.0, last):
            break
        last = inertia
    labels = np.asarray(_curate.kmeans_assign_labels(flat, n, dim, [c.tolist() for c in centroids]), dtype="int64")
    return centroids, labels


def kmeans_distributed(
    table: Any,
    column: str = "embedding",
    *,
    k: int = 256,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 0,
    runner: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distributed k-means: each shard runs the Rust assign+accumulate map on a
    worker; the driver merges partial (sum, count) into new centroids and
    iterates. Returns (centroids, labels). Scales k-means past one machine.
    """
    import jude

    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    from jude.runners import _ray_shim as shim

    con = jude.connect()
    tbl = table if isinstance(table, pa.Table) else table.to_arrow()
    rel = con.from_arrow(tbl)
    parts = r._partition_tables(rel)
    workers = r._ensure_workers()
    dim = len(tbl.column(column)[0].as_py())

    # seed centroids from a sample on the driver
    mat0 = _as_matrix(tbl.slice(0, min(tbl.num_rows, max(k * 4, 1000))), column)
    centroids = _init_centroids(mat0, min(k, mat0.shape[0]), seed)

    last = float("inf")
    for _ in range(max_iter):
        cents_list = [c.tolist() for c in centroids]
        refs = [
            workers[r.mgr.worker_for(i)].kmeans_map.remote(part, column, cents_list)
            for i, part in enumerate(parts)
        ]
        partials = [p for p in shim.get(refs) if p is not None]
        # merge partial sums/counts across shards
        kk = len(centroids)
        sums = np.zeros((kk, dim), dtype="float64")
        counts = np.zeros(kk, dtype="int64")
        inertia = 0.0
        for psum, pcount, pin in partials:
            sums += np.asarray(psum, dtype="float64")
            counts += np.asarray(pcount, dtype="int64")
            inertia += pin
        for c in range(kk):
            if counts[c] > 0:
                centroids[c] = (sums[c] / counts[c]).astype("float32")
        if last - inertia <= tol * max(1.0, last):
            break
        last = inertia

    # final labels (single-node assign over the whole table is cheap: just a
    # nearest-centroid scan)
    mat = _as_matrix(tbl, column)
    labels = np.asarray(
        _curate.kmeans_assign_labels(mat.reshape(-1).tolist(), mat.shape[0], dim, [c.tolist() for c in centroids]),
        dtype="int64",
    )
    return centroids, labels


def semantic_dedup_clustered(
    table: Any,
    column: str = "embedding",
    *,
    n_clusters: int = 256,
    threshold: float = 0.9,
    max_iter: int = 15,
    seed: int = 0,
    keep_cluster: bool = False,
) -> pa.Table:
    """Scalable semantic dedup: k-means cluster the embeddings, then run the
    O(n^2) cosine dedup only WITHIN each cluster (near-dups share a cluster), so
    total cost is ~sum of per-cluster n_c^2 << global n^2. Approximates the exact
    semantic_dedup at a fraction of the cost — the SemDeDup approach used on
    large corpora.
    """
    from jude import curate as _c

    tbl = table if isinstance(table, pa.Table) else table.to_arrow()
    n = tbl.num_rows
    if n == 0:
        return tbl
    _, labels = kmeans(tbl, column, k=min(n_clusters, n), max_iter=max_iter, seed=seed)
    # within each cluster, run exact semantic dedup and collect survivors
    keep_mask = np.zeros(n, dtype=bool)
    cluster_rep = np.arange(n, dtype="int64")
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        if len(idx) == 1:
            keep_mask[idx[0]] = True
            continue
        sub = tbl.take(pa.array(idx.tolist(), type=pa.int64()))
        rep = _c.semantic_dedup(sub, embedding_column=column, threshold=threshold, keep_cluster=True)
        reps = rep.column("sem_cluster").to_pylist()  # local reps (0-based within sub)
        for local_i, r_local in enumerate(reps):
            gi = idx[local_i]
            grep = idx[r_local]
            cluster_rep[gi] = grep
            if r_local == local_i:
                keep_mask[gi] = True
    if keep_cluster:
        return tbl.append_column("sem_cluster", pa.array(cluster_rep.tolist(), type=pa.int64()))
    keep = np.nonzero(keep_mask)[0]
    return tbl.take(pa.array(keep.tolist(), type=pa.int64()))

"""jude.curate_dist — distributed data-curation over the Ray runner.

The distributed forms of jude.curate / jude.curate_mm. Two shapes:

- **map-style** (embarrassingly parallel): partition the table, apply the pure
  curator to each shard on a worker, concat. Used for chunk / content-hash /
  quality / language / image-quality — anything whose per-row result is
  independent. ``dist_map(runner, table, op, **kwargs)``.

- **dedup shuffle** (same key must co-locate): exact dedup shuffles rows by
  content hash; fuzzy dedup computes MinHash signatures, routes by LSH band key,
  then each reducer verifies candidate pairs and union-finds a duplicate cluster.
  ``dist_exact_dedup`` / ``dist_fuzzy_dedup``.

Scheduling (partition sizing, worker assignment, bucket count) is the Rust
WorkerManager's, same as every other distributed op; this module only wires the
curators onto it. Correctness matches the single-node jude.curate functions.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__ = [
    "dist_map",
    "dist_chunk_text",
    "dist_quality_filter",
    "dist_detect_language",
    "dist_language_filter",
    "dist_add_content_hash",
    "dist_image_quality_filter",
    "dist_exact_dedup",
    "dist_fuzzy_dedup",
    "dist_global_shuffle",
    "dist_blend_datasets",
]

# ops that are embarrassingly parallel (per-row independent) -> map-style
_MAP_OPS = {
    "chunk_text", "add_content_hash", "quality_filter", "quality_signals",
    "detect_language", "language_filter", "add_image_quality",
    "image_quality_filter", "add_image_hash",
}


def _runner(runner: Any = None) -> Any:
    if runner is not None:
        return runner
    from jude.runners import get_or_create_runner

    return get_or_create_runner()


class _UnionFind:
    """Disjoint-set with path compression and **union-by-min**: a component's
    root is always its smallest member index. Fed incrementally so the driver
    never holds the whole edge set (L2.1). Union-by-min makes the survivor the
    lowest row index, matching single-node fuzzy_dedup's keep-lowest semantics."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo  # smaller index becomes the root


def dist_map(table: pa.Table, op: str, *, runner: Any = None, **kwargs: Any) -> pa.Table:
    """Apply a map-style curator to a table, distributed: partition -> per-shard
    apply on workers -> concat. `op` is a jude.curate/curate_mm function name."""
    if op not in _MAP_OPS:
        raise ValueError(f"{op!r} is not a map-style curator; use dist_exact_dedup/dist_fuzzy_dedup")
    r = _runner(runner)
    import jude

    con = jude.connect()
    rel = con.from_arrow(table)
    parts = r._partition_tables(rel)
    workers = r._ensure_workers()
    submit = [
        (lambda i=i, part=part: workers[r.mgr.worker_for(i)].curate_map.remote(part, op, dict(kwargs)))
        for i, part in enumerate(parts)
    ]
    outs = [t for t in r._dispatch_bounded(submit) if t is not None and t.num_rows > 0]
    if not outs:
        # apply once locally to get the right (possibly widened) schema for empties
        from jude import curate, curate_mm

        fn = getattr(curate, op, None) or getattr(curate_mm, op, None)
        return fn(table.slice(0, 0), **kwargs)
    return pa.concat_tables(outs).combine_chunks()


# --- map-style convenience wrappers -----------------------------------------


def dist_chunk_text(table, *, runner=None, **kw):
    return dist_map(table, "chunk_text", runner=runner, **kw)


def dist_quality_filter(table, *, runner=None, **kw):
    return dist_map(table, "quality_filter", runner=runner, **kw)


def dist_detect_language(table, *, runner=None, **kw):
    return dist_map(table, "detect_language", runner=runner, **kw)


def dist_language_filter(table, *, runner=None, **kw):
    return dist_map(table, "language_filter", runner=runner, **kw)


def dist_add_content_hash(table, *, runner=None, **kw):
    return dist_map(table, "add_content_hash", runner=runner, **kw)


def dist_image_quality_filter(table, *, runner=None, **kw):
    return dist_map(table, "image_quality_filter", runner=runner, **kw)


# --- dedup shuffle -----------------------------------------------------------


def dist_exact_dedup(
    table: pa.Table, *, column: str = "text", normalize: bool = True,
    keep_hash: bool = False, runner: Any = None,
) -> pa.Table:
    """Distributed exact dedup: shuffle rows by content hash so identical docs
    co-locate, drop duplicates per bucket. Result matches curate.exact_dedup."""
    from jude.runners import _ray_shim as shim
    import jude

    r = _runner(runner)
    con = jude.connect()
    parts = r._partition_tables(con.from_arrow(table))
    workers = r._ensure_workers()
    b = r.mgr.shuffle_bucket_count(None)
    bucket_workers = r.mgr.shuffle_bucket_workers(None)
    # producer: hash + route to buckets
    refs = [
        workers[r.mgr.worker_for(i)].curate_hash_bucketize.options(num_returns=b).remote(part, column, normalize, b)
        for i, part in enumerate(parts)
    ]
    refs = [x if isinstance(x, list) else [x] for x in refs]
    # reducer: per-bucket dedup
    out = [
        workers[bucket_workers[bkt]].curate_exact_dedup_bucket.remote(
            [refs[p][bkt] for p in range(len(parts))], keep_hash
        )
        for bkt in range(b)
    ]
    tables = [t for t in shim.get(out) if t is not None and t.num_rows > 0]
    if not tables:
        return table.slice(0, 0)
    return pa.concat_tables(tables).combine_chunks()


def dist_fuzzy_dedup(
    table: pa.Table, *, column: str = "text", threshold: float = 0.7,
    num_hashes: int = 128, ngram: int = 2, bands: int | None = None, seed: int = 1,
    keep_cluster: bool = False, cc_workers: int = 0, runner: Any = None,
) -> pa.Table:
    """Distributed MinHash-LSH fuzzy dedup, recall-matched to single-node.

    Each shard computes MinHash signatures and routes every row to a bucket for
    EACH of its LSH band keys (near-dups sharing ANY band co-locate — not just
    the first band). Each reducer verifies candidate pairs by Jaccard>=threshold
    and emits near-dup EDGES; the driver runs ONE global union-find over all
    edges so clusters that span buckets (A~B in one bucket, B~C in another) merge
    correctly, then keeps one row per cluster. This matches ``curate.fuzzy_dedup``
    semantics at scale (the old first-band routing silently lost recall). Only
    (row-id, band-key, signature) is shuffled, never the full row.
    """
    from jude.runners import _ray_shim as shim
    import jude
    from jude.jude import _curate
    from jude.curate import optimal_lsh_bands

    if bands is None:
        bands = optimal_lsh_bands(threshold, num_hashes)  # calibrate to threshold (C3)
    r = _runner(runner)
    con = jude.connect()
    parts = r._partition_tables(con.from_arrow(table))
    workers = r._ensure_workers()
    b = r.mgr.shuffle_bucket_count(None)
    bucket_workers = r.mgr.shuffle_bucket_workers(None)
    # global row-id offsets: parts are in-order slices of `table`, so row
    # (offset[p] + i) of the corpus is row i of part p — lets the driver map
    # edges back to original rows.
    offsets: list[int] = []
    acc = 0
    for part in parts:
        offsets.append(acc)
        acc += part.num_rows
    refs = [
        workers[r.mgr.worker_for(i)].curate_minhash_edges.options(num_returns=b).remote(
            part, column, num_hashes, ngram, bands, seed, b, offsets[i]
        )
        for i, part in enumerate(parts)
    ]
    refs = [x if isinstance(x, list) else [x] for x in refs]
    edge_refs = [
        workers[bucket_workers[bkt]].curate_fuzzy_edges_bucket.remote(
            [refs[p][bkt] for p in range(len(parts))], threshold
        )
        for bkt in range(b)
    ]
    n = table.num_rows
    if cc_workers and cc_workers > 0:
        # L2.1 (trillion-scale): distribute the LABEL ARRAY too — label
        # propagation across cc_workers actors, so the driver never holds n
        # labels, only the sparse {rid -> smaller-rep} map for merged rows. Edges
        # stay as refs (not materialized on the driver).
        from jude.dist_cc import connected_components

        label = connected_components(edge_refs, num_workers=cc_workers)
        if keep_cluster:
            reps = [label.get(i, i) for i in range(n)]
            return table.append_column("dup_cluster", pa.array(reps, type=pa.int64())).combine_chunks()
        # a rid survives iff it is a representative: absent from `label` (singleton
        # or component-min) OR maps to itself.
        keep = [i for i in range(n) if label.get(i, i) == i]
        if len(keep) == n:
            return table
        return table.take(pa.array(keep, type=pa.int64())).combine_chunks()
    # L2.1 (default): stream each bucket's edges into an incremental union-find
    # instead of collecting EVERY edge on the driver then running one CC pass.
    # Peak driver memory is the label array (n) + one bucket's edges at a time,
    # not the whole edge set — so a high-dup corpus with a huge edge count doesn't
    # blow the driver. union-by-min keeps the lowest row index as each cluster's
    # representative (matches single-node fuzzy_dedup's keep-lowest semantics).
    uf = _UnionFind(n)
    for ref in edge_refs:
        et = shim.get([ref])[0]           # resolve one bucket at a time (streamed)
        if et is None or et.num_rows == 0:
            continue
        aa = et.column("a").to_pylist()
        bb = et.column("b").to_pylist()
        for a, b in zip(aa, bb):
            uf.union(a, b)
        del et, aa, bb                    # free this bucket's edges before the next
    reps = [uf.find(i) for i in range(n)]
    if keep_cluster:
        return table.append_column("dup_cluster", pa.array(reps, type=pa.int64())).combine_chunks()
    keep = [i for i in range(n) if reps[i] == i]
    if len(keep) == n:
        return table
    return table.take(pa.array(keep, type=pa.int64())).combine_chunks()


# --- C9. distributed global shuffle + blend ---------------------------------


def dist_global_shuffle(table: pa.Table, *, seed: int = 0, runner: Any = None) -> pa.Table:
    """Distributed global shuffle: scatter each partition's rows to random output
    buckets (rows from all partitions interleave), then each reducer concats its
    bucket and permutes locally. The result is globally random without ever
    materializing the whole table on one worker — the training-data shuffle that
    scales past one machine's memory.
    """
    from jude.runners import _ray_shim as shim
    import jude

    r = _runner(runner)
    con = jude.connect()
    parts = r._partition_tables(con.from_arrow(table))
    workers = r._ensure_workers()
    b = r.mgr.shuffle_bucket_count(None)
    bucket_workers = r.mgr.shuffle_bucket_workers(None)
    # producer: scatter each partition to random buckets
    refs = [
        workers[r.mgr.worker_for(i)].curate_shuffle_scatter.options(num_returns=b).remote(part, b, seed, i)
        for i, part in enumerate(parts)
    ]
    refs = [x if isinstance(x, list) else [x] for x in refs]
    # reducer: gather + local permute per bucket
    out = [
        workers[bucket_workers[bkt]].curate_shuffle_gather.remote(
            [refs[p][bkt] for p in range(len(parts))], seed, bkt
        )
        for bkt in range(b)
    ]
    tables = [t for t in shim.get(out) if t is not None and t.num_rows > 0]
    if not tables:
        return table.slice(0, 0)
    # buckets are already internally shuffled; concat preserves global randomness
    return pa.concat_tables(tables).combine_chunks()


def dist_blend_datasets(
    tables: list, weights: list | None = None, *, total_rows: int | None = None,
    seed: int = 0, runner: Any = None,
) -> pa.Table:
    """Distributed dataset blend: sample each source to its weighted quota (via
    the single-node blend, which is cheap index math), concat, then distributed
    global-shuffle the result so the mix is globally random across the cluster.
    """
    from jude import curate as _c

    blended = _c.blend_datasets(tables, weights, total_rows=total_rows, seed=seed)
    return dist_global_shuffle(blended, seed=seed, runner=runner)

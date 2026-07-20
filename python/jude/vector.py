"""jude.vector — in-process vector search over DuckDB's native functions + VSS.

jude's positioning as a large-model data engine needs first-class vector ops
for embeddings: KNN, similarity scoring, dedup by cosine. DuckDB (stock, which
jude embeds) already provides these — this module surfaces them as a clean API
instead of raw SQL, and wires the optional VSS extension (HNSW index) which the
connection auto-loads on first use.

Two tiers, complementary to the persistent Lance vector index:
- **native functions** (no extension): array_cosine_distance / array_distance /
  array_inner_product — exact brute-force KNN + similarity columns. Best for
  small/medium in-memory embedding tables and rerank.
- **VSS HNSW** (auto-loaded extension): a persistent-ish approximate index for
  larger tables.

For TB-scale persistent ANN, use the Lance vector index (jude._lance) instead;
this is the lightweight in-process tier.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__ = ["knn", "add_similarity", "distributed_knn", "create_hnsw_index", "METRICS",
           "recall_at_k", "knn_rerank", "knn_ann_resident", "range_search", "mmr",
           "high_recall_knn", "distributed_ann_knn", "distributed_knn_resident",
           "distributed_knn_resident_batch", "distributed_ann_knn_routed"]

# metric -> (distance function, ascending?) ; smaller distance = nearer
METRICS = {
    "cosine": ("array_cosine_distance", True),
    "l2": ("array_distance", True),
    "l2sq": ("array_distance", True),
    "ip": ("array_negative_inner_product", True),  # MIPS via negative IP, ascending
    "inner": ("array_negative_inner_product", True),
}


def _vec_literal(query: list[float]) -> str:
    dim = len(query)
    inner = ", ".join(repr(float(x)) for x in query)
    return f"[{inner}]::FLOAT[{dim}]"


def knn(
    con: Any,
    table: str,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    metric: str = "cosine",
    select: str = "*",
    where: str | None = None,
) -> pa.Table:
    """Exact k-nearest-neighbor search over an embedding column via DuckDB's
    native array distance functions (no index needed). ``table`` is a registered
    table/view name; ``column`` is a ``FLOAT[dim]`` (or list) embedding column.

    Returns the top-k rows plus a ``_distance`` column, nearest first.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(METRICS)}")
    fn, _asc = METRICS[metric]
    qlit = _vec_literal(query)
    filt = f" WHERE {where}" if where else ""
    sql = (
        f"SELECT {select}, {fn}({column}, {qlit}) AS _distance "
        f"FROM {table}{filt} ORDER BY _distance ASC LIMIT {int(k)}"
    )
    return con.sql(sql).to_arrow()


def add_similarity(
    con: Any,
    table: str,
    column: str,
    query: list[float],
    *,
    metric: str = "cosine",
    out_column: str = "similarity",
) -> pa.Table:
    """Add a similarity/distance column for every row against ``query`` (no
    ordering/limit) — for thresholded filtering or scoring. Cosine returns
    similarity (1 - distance); others return the distance."""
    qlit = _vec_literal(query)
    if metric == "cosine":
        expr = f"array_cosine_similarity({column}, {qlit})"
    elif metric in ("ip", "inner"):
        expr = f"array_inner_product({column}, {qlit})"
    else:
        expr = f"array_distance({column}, {qlit})"
    return con.sql(f"SELECT *, {expr} AS {out_column} FROM {table}").to_arrow()


def distributed_knn(
    table: Any,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    metric: str = "cosine",
    runner: Any = None,
) -> "pa.Table":
    """Distributed exact k-NN. DuckDB's native array-distance KNN is single-
    machine; jude scales it: each worker computes its local top-k over its shard
    (map), then the driver merges the local top-ks and takes the global top-k.

    ``table`` is a pyarrow Table / jude Relation whose ``column`` holds the
    embeddings. Returns the global top-k rows + a ``_distance`` column. Answer to
    "does DuckDB native vector support distributed search": not natively — this
    is how jude makes it distributed.
    """
    import jude

    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(METRICS)}")
    fn, _asc = METRICS[metric]
    qlit = _vec_literal(query)
    knn_sql = f"SELECT *, {fn}({column}, {qlit}) AS _distance FROM part ORDER BY _distance ASC LIMIT {int(k)}"

    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    con = jude.connect()
    rel = con.from_arrow(table if isinstance(table, pa.Table) else table.to_arrow())
    parts = r._partition_tables(rel)
    workers = r._ensure_workers()
    submit = [
        (lambda i=i, part=part: workers[r.mgr.worker_for(i)].run_sql_on_table.remote(part, knn_sql))
        for i, part in enumerate(parts)
    ]
    locals_ = [t for t in r._dispatch_bounded(submit) if t is not None and t.num_rows > 0]
    if not locals_:
        return rel.to_arrow().slice(0, 0)
    merged = pa.concat_tables(locals_).combine_chunks()
    con2 = jude.connect()
    con2.register("_m", merged)
    return con2.sql(f"SELECT * FROM _m ORDER BY _distance ASC LIMIT {int(k)}").to_arrow()


def create_hnsw_index(con: Any, table: str, column: str, *, metric: str = "cosine", name: str | None = None) -> None:
    """Create a VSS HNSW index on an embedding column for approximate KNN. The
    connection auto-loads the ``vss`` extension. On-disk HNSW persistence is
    experimental in DuckDB (a persistent DB needs
    ``SET hnsw_enable_experimental_persistence=true``, set here best-effort);
    in-memory HNSW is stable. For large persistent ANN, prefer the Lance index.
    """
    idx = name or f"{table}_{column}_hnsw"
    vss_metric = {"cosine": "cosine", "l2": "l2sq", "l2sq": "l2sq", "ip": "ip", "inner": "ip"}.get(metric, "cosine")
    con.execute("LOAD vss")
    con.execute("SET hnsw_enable_experimental_persistence=true")
    con.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {table} USING HNSW ({column}) WITH (metric='{vss_metric}')")


# --- high-recall large-k retrieval (top-100k) --------------------------------


def recall_at_k(approx_ids: list, exact_ids: list, k: int | None = None) -> float:
    """Recall@k = |approx ∩ exact| / |exact|, over the top-k of each. The metric
    to quantify how much an ANN result misses vs the exact ground truth — run it
    to tune nprobes/refine until recall meets your target."""
    kk = k if k is not None else len(exact_ids)
    a = set(approx_ids[:kk])
    e = set(exact_ids[:kk])
    if not e:
        return 1.0
    return len(a & e) / len(e)


def knn_rerank(
    path: str,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    overfetch: int = 4,
    nprobes: int | None = None,
    metric: str = "cosine",
    columns: Any = None,
    id_column: str = "id",
    where: str | None = None,
) -> "pa.Table":
    """Two-stage high-recall ANN over a Lance dataset: fetch ``k * overfetch``
    approximate candidates (scan more IVF cells via ``nprobes``), then RE-RANK
    them with EXACT distances and keep the true top-k. Over-fetch + exact re-rank
    is the standard way to push ANN recall toward 100% at a fraction of a full
    brute-force scan — the more you over-fetch, the higher the recall.

    Requires a Lance vector index on ``column`` (Connection.create_lance_vector_index).
    """
    from jude import _lance

    cand_k = max(k, k * max(1, overfetch))
    # stage 1: ANN over-fetch — fetch candidates WITH their vectors (reuse the
    # cached dataset handle; no reopen per query).
    ds = _lance.dataset_cached(path)
    want = None
    if columns is not None:
        want = list(dict.fromkeys(list(columns) + [column]))
    nearest = {"column": column, "q": [float(x) for x in query], "k": cand_k}
    if nprobes is not None:
        nearest["nprobes"] = int(nprobes)
    # filtered ANN: push a metadata predicate INTO the index scan (pre-filter) so
    # the top-k is over rows matching `where` — the common RAG pattern
    # "similar AND category=... AND date>...". Lance evaluates the filter during
    # the IVF scan (prefilter=True), not after, so recall isn't lost to filtering.
    kw = {"nearest": nearest, "columns": want}
    if where:
        kw["filter"] = where
        kw["prefilter"] = True
    cands = ds.to_table(**kw)
    if cands.num_rows <= k:
        return cands
    # stage 2: exact re-rank the candidates in numpy — no DuckDB connection per
    # query (that was the dominant overhead). Decode the candidate vectors via a
    # fast Arrow->numpy path (flatten the FixedSizeList child buffer); a per-query
    # .to_pylist() here materializes k*overfetch*dim Python floats and dominated
    # latency at realistic dims (it made this ~10x slower than the Lance fetch).
    import numpy as np

    col = cands.column(column).combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        vecs = (col.flatten().to_numpy(zero_copy_only=False)
                .astype("float32", copy=False).reshape(cands.num_rows, col.type.list_size))
    else:
        vecs = np.asarray(col.to_pylist(), dtype="float32")
    qv = np.asarray(query, dtype="float32")
    if metric == "cosine":
        vn = np.linalg.norm(vecs, axis=1)
        qn = np.linalg.norm(qv) or 1.0
        sims = (vecs @ qv) / (np.where(vn == 0, 1.0, vn) * qn)
        dist = 1.0 - sims
    elif metric in ("ip", "inner"):
        dist = -(vecs @ qv)
    else:  # l2 / l2sq
        diff = vecs - qv
        dist = np.einsum("ij,ij->i", diff, diff)
    order = np.argsort(dist)[:k]
    out = cands.take(pa.array(order.tolist(), type=pa.int64()))
    if "_distance" in out.column_names:
        out = out.drop_columns(["_distance"])
    return out.append_column("_distance", pa.array(dist[order].tolist(), type=pa.float64()))


# --- in-memory-rerank ANN: IVF prunes, RAM reranks (the fast path) -----------

_RESIDENT_VEC: dict = {}  # (path, column) -> (ids, id_to_row, matrix, norms)


def _resident_vectors(path: str, column: str):
    """Load a Lance dataset's id + vector column into RAM once and cache it, as
    (ids, id->row map, float32 matrix, row norms). Reused across queries so the
    exact re-rank never re-reads vectors from Lance."""
    import numpy as np

    key = (path, column)
    r = _RESIDENT_VEC.get(key)
    if r is None:
        from jude import _lance

        tbl = _lance.dataset_cached(path).to_table(columns=None)
        ids = np.asarray(tbl.column("id").to_numpy(zero_copy_only=False))
        col = tbl.column(column).combine_chunks()
        d = col.type.list_size
        mat = col.flatten().to_numpy(zero_copy_only=False).astype("float32", copy=False).reshape(len(ids), d)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        id_to_row = {int(x): i for i, x in enumerate(ids)}
        r = (ids, id_to_row, mat, norms)
        _RESIDENT_VEC[key] = r
    return r


def knn_ann_resident(
    path: str,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    overfetch: int = 5,
    nprobes: int | None = None,
    metric: str = "cosine",
) -> "pa.Table":
    """FAST two-stage ANN: Lance IVF returns candidate **IDs only** (no vector
    materialization through Arrow — that per-query fetch is what makes plain
    ``knn_rerank`` slow at high dim), then re-rank the candidates with EXACT
    distances against an **in-RAM resident matrix** (loaded + cached once). Same
    recall as ``knn_rerank`` (identical candidate set + exact re-rank) but a
    fraction of the latency because the vectors never leave RAM per query.

    Requires a Lance vector index on ``column``; the full column is held resident
    in memory (see ``_resident_vectors``) — for datasets that fit in RAM. For
    larger-than-RAM corpora use ``knn_rerank`` (streams vectors from Lance) or the
    distributed sharded path.
    """
    import numpy as np

    from jude import _lance

    ids, id_to_row, mat, norms = _resident_vectors(path, column)
    cand_k = max(k, k * max(1, overfetch))
    ds = _lance.dataset_cached(path)
    nearest = {"column": column, "q": [float(x) for x in query], "k": cand_k}
    if nprobes is not None:
        nearest["nprobes"] = int(nprobes)
    # stage 1: candidate IDs only — no vector column materialized to Python
    cand = ds.to_table(nearest=nearest, columns=["id"])
    cand_ids = cand.column("id").to_numpy(zero_copy_only=False)
    rows = np.fromiter((id_to_row.get(int(x), -1) for x in cand_ids), dtype=np.int64,
                       count=len(cand_ids))
    rows = rows[rows >= 0]
    if rows.size == 0:
        return pa.table({"id": pa.array([], type=pa.int64()),
                         "_distance": pa.array([], type=pa.float64())})
    # stage 2: exact re-rank against the RAM matrix
    sub = mat[rows]
    qv = np.asarray(query, dtype="float32")
    if metric == "cosine":
        qn = np.linalg.norm(qv) or 1.0
        dist = 1.0 - (sub @ qv) / (norms[rows] * qn)
    elif metric in ("ip", "inner"):
        dist = -(sub @ qv)
    else:
        diff = sub - qv
        dist = np.einsum("ij,ij->i", diff, diff)
    kk = min(k, dist.shape[0])
    part = np.argpartition(dist, kk - 1)[:kk]
    order = part[np.argsort(dist[part])]
    keep = rows[order]
    return pa.table({"id": pa.array(ids[keep].tolist(), type=pa.int64()),
                     "_distance": pa.array(dist[order].tolist(), type=pa.float64())})


# --- range/threshold search + MMR diversification ----------------------------


def range_search(
    con: Any,
    table: str,
    column: str,
    query: list[float],
    *,
    radius: float,
    metric: str = "cosine",
    select: str = "*",
    where: str | None = None,
    limit: int | None = None,
) -> "pa.Table":
    """Threshold / range search: return EVERY row whose distance to ``query`` is
    ``<= radius`` (not a fixed top-k). This is the primitive for dedup / near-
    duplicate detection / entity resolution ("everything within ε"), where the
    number of matches is unknown up front. Exact (DuckDB native distance). For
    cosine, ``radius`` is a cosine *distance* (1 - similarity), e.g. 0.1 ≈ sim≥0.9.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(METRICS)}")
    fn, _asc = METRICS[metric]
    qlit = _vec_literal(query)
    filt = f" AND ({where})" if where else ""
    lim = f" LIMIT {int(limit)}" if limit else ""
    sql = (
        f"SELECT {select}, {fn}({column}, {qlit}) AS _distance FROM {table} "
        f"WHERE {fn}({column}, {qlit}) <= {float(radius)}{filt} "
        f"ORDER BY _distance ASC{lim}"
    )
    return con.sql(sql).to_arrow()


def mmr(
    candidates: "pa.Table",
    column: str,
    query: list[float],
    *,
    k: int = 10,
    lambda_: float = 0.5,
    metric: str = "cosine",
) -> "pa.Table":
    """Maximal Marginal Relevance re-ranking: greedily pick ``k`` rows that are
    relevant to ``query`` yet diverse from each other, trading the two off with
    ``lambda_`` (1.0 = pure relevance, 0.0 = pure diversity). ``candidates`` is a
    small table (e.g. the top-N from a KNN) holding the ``column`` embeddings.

    MMR score for a candidate c already-having-selected S:
        lambda * sim(c, query) - (1 - lambda) * max_{s in S} sim(c, s)
    The RAG primitive for de-redundant context selection.
    """
    import numpy as np

    col = candidates.column(column).combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        vecs = (col.flatten().to_numpy(zero_copy_only=False)
                .astype("float32", copy=False).reshape(candidates.num_rows, col.type.list_size))
    else:
        vecs = np.asarray(col.to_pylist(), dtype="float32")
    n = vecs.shape[0]
    if n == 0:
        return candidates
    qv = np.asarray(query, dtype="float32")

    def sim(a, b):  # cosine similarity (works well for all metrics as a proxy)
        an = np.linalg.norm(a, axis=-1)
        bn = np.linalg.norm(b) or 1.0
        return (a @ b) / (np.where(an == 0, 1.0, an) * bn)

    rel = sim(vecs, qv)  # relevance to query
    selected: list[int] = []
    remaining = set(range(n))
    kk = min(k, n)
    for _ in range(kk):
        best, best_score = -1, -1e18
        for c in remaining:
            div = 0.0 if not selected else max(float(sim(vecs[c:c + 1], vecs[s])[0]) for s in selected)
            score = lambda_ * float(rel[c]) - (1.0 - lambda_) * div
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.discard(best)
    return candidates.take(pa.array(selected, type=pa.int64()))



def high_recall_knn(
    table: Any,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    metric: str = "cosine",
    runner: Any = None,
    prefer_exact_below: int = 5_000_000,
) -> "pa.Table":
    """Retrieve the top-k (supports very large k, e.g. 100,000) with recall as
    high as possible. Strategy:

    - For datasets up to ``prefer_exact_below`` rows, use **exact** distributed
      brute-force (``distributed_knn``) — 100% recall, fully parallel. For a
      large k like 100k this is usually both the highest-recall AND the most
      practical choice, since ANN indexes are tuned for tiny-k and degrade badly
      when k is a large fraction of the dataset.
    - For bigger datasets, exact is still correct but you may prefer the indexed
      two-stage ``knn_rerank`` (over-fetch ANN + exact re-rank) on a Lance
      dataset for speed; call it directly with a path.

    Returns the top-k rows + a ``_distance`` column, nearest first.
    """
    tbl = table if isinstance(table, pa.Table) else table.to_arrow()
    # exact distributed brute-force — the 100%-recall path (map local top-k ->
    # merge global top-k), which scales to large k and large row counts.
    return distributed_knn(tbl, column, query, k=k, metric=metric, runner=runner)


# --- billion-scale: distributed sharded ANN ---------------------------------


def distributed_ann_knn(
    shard_paths: list,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    overfetch: int = 3,
    nprobes: int | None = None,
    metric: str = "cosine",
    id_column: str = "id",
    where: str | None = None,
    runner: Any = None,
) -> "pa.Table":
    """Distributed sharded ANN — the billion-scale retrieval algorithm.

    Each element of ``shard_paths`` is a pre-indexed Lance dataset (one shard of
    the corpus, its own IVF index). The query fans out to all shards in
    parallel; each shard returns its LOCAL top-``k`` via two-stage ANN
    (over-fetch + exact re-rank against its index); the driver merges the shard
    top-ks and takes the GLOBAL top-``k``.

    Why this for 1B vectors -> top-1M: brute force over 1B vectors (~TBs) is
    infeasible on one machine and slow even distributed; a single monolithic
    index doesn't fit either. Sharding + per-shard ANN keeps each index in one
    machine's memory, searches all shards concurrently, and merges — the
    architecture Milvus/Vespa use at scale. Recall is tuned per shard by
    ``overfetch``/``nprobes`` (higher = closer to exact).

    Correctness note: merging per-shard top-k gives the exact global top-k *if*
    each shard's ANN recalls its true local top-k. With ANN that's approximate,
    so raise overfetch/nprobes (or use IVF_FLAT shards) to hit a recall target;
    measure with ``recall_at_k``.
    """
    import numpy as np

    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    workers = r._ensure_workers()
    # fan out: one shard per worker (round-robin), local top-k each (with an
    # optional `where` metadata pre-filter pushed into each shard's index scan)
    refs = [
        workers[r.mgr.worker_for(i)].vector_knn_shard.remote(
            path, column, list(query), k, overfetch, nprobes, metric, where
        )
        for i, path in enumerate(shard_paths)
    ]
    from jude.runners import _ray_shim as shim

    locals_ = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
    if not locals_:
        return pa.table({})
    # driver merge: global top-k over the union of shard top-ks, in numpy — no
    # per-query DuckDB connection (that reopen was a large fixed cost per query).
    merged = pa.concat_tables(locals_).combine_chunks()
    d = merged.column("_distance").to_numpy(zero_copy_only=False)
    order = np.argsort(d, kind="stable")[: int(k)]
    return merged.take(pa.array(order.tolist(), type=pa.int64()))


def distributed_knn_resident(
    shard_paths: list,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    metric: str = "cosine",
    runner: Any = None,
) -> "pa.Table":
    """Distributed EXACT KNN over RESIDENT shards. Each element of ``shard_paths``
    is a persistent Lance dataset (a shard of the corpus, no index). A query fans
    out to all shards; each worker exact-scans its (cached, resident) shard and
    returns its local top-k; the driver merges to the global exact top-k (100%
    recall). Unlike ``distributed_knn`` (which re-partitions and re-ships the
    whole table from the driver every query — fine for one-shot, terrible for
    repeated queries), here the data stays resident on the workers and only the
    query vector travels — the correct architecture for repeated vector search.
    """
    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    from jude.runners import _ray_shim as shim

    workers = r._ensure_workers()
    refs = [
        workers[r.mgr.worker_for(i)].vector_exact_shard.remote(path, column, list(query), k, metric)
        for i, path in enumerate(shard_paths)
    ]
    locals_ = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
    if not locals_:
        return pa.table({})
    # merge W partial top-ks in numpy — no per-query DuckDB roundtrip (that was
    # a fixed cost that throttled worker scaling).
    import numpy as np

    merged = pa.concat_tables(locals_).combine_chunks()
    d = merged.column("_distance").to_numpy(zero_copy_only=False)
    order = np.argsort(d, kind="stable")[: int(k)]
    return merged.take(pa.array(order.tolist(), type=pa.int64()))


def distributed_knn_resident_batch(
    shard_paths: list,
    column: str,
    queries: list,
    *,
    k: int = 10,
    metric: str = "cosine",
    runner: Any = None,
) -> list:
    """BATCHED distributed EXACT KNN over resident shards — the throughput path.

    ``queries`` is a list of query vectors. The whole batch is sent to each
    worker in a SINGLE RPC; each worker scores the entire batch against its
    resident shard with one BLAS GEMM (B x N_shard) and returns B local top-ks.
    The driver merges per-query across workers. This amortizes RPC + merge over
    the batch, so aggregate QPS scales near-linearly with workers — unlike
    per-query fan-out (``distributed_knn_resident``), which pays fixed RPC
    overhead per query and plateaus.

    Returns a list of ``pa.Table`` (one per query, id + _distance, nearest first).
    """
    import numpy as np

    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    from jude.runners import _ray_shim as shim

    qs = [list(q) for q in queries]
    workers = r._ensure_workers()
    refs = [
        workers[r.mgr.worker_for(i)].vector_exact_shard_batch.remote(path, column, qs, k, metric)
        for i, path in enumerate(shard_paths)
    ]
    parts = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
    out: list = [pa.table({"id": pa.array([], type=pa.int64()),
                           "_distance": pa.array([], type=pa.float64())}) for _ in qs]
    if not parts:
        return out
    merged = pa.concat_tables(parts).combine_chunks()
    qi = merged.column("qi").to_numpy(zero_copy_only=False)
    dist = merged.column("_distance").to_numpy(zero_copy_only=False)
    ids = merged.column("id").to_numpy(zero_copy_only=False)
    for i in range(len(qs)):
        sel = np.where(qi == i)[0]
        if sel.size == 0:
            continue
        di = dist[sel]
        order = sel[np.argsort(di, kind="stable")[: int(k)]]
        out[i] = pa.table({
            "id": pa.array(ids[order].tolist(), type=pa.int64()),
            "_distance": pa.array(dist[order].tolist(), type=pa.float64()),
        })
    return out


# --- cluster-routed distributed ANN: query touches only relevant shards ------


def distributed_ann_knn_routed(
    shard_paths: list,
    shard_centroids: list,
    column: str,
    query: list[float],
    *,
    k: int = 10,
    n_shards_probe: int = 2,
    overfetch: int = 3,
    nprobes: int | None = None,
    metric: str = "cosine",
    where: str | None = None,
    runner: Any = None,
) -> "pa.Table":
    """CLUSTER-ROUTED distributed ANN — the true billion-scale architecture.

    Unlike ``distributed_ann_knn`` (which fans EVERY query out to ALL shards),
    here the corpus is partitioned by cluster: ``shard_centroids[i]`` is the
    centroid (mean vector) of shard ``i``. A query is routed ONLY to the
    ``n_shards_probe`` shards whose centroid is nearest the query — so it touches
    a few shards, not W. This is how Milvus/Vespa keep query cost sub-linear in
    cluster size: coarse routing on centroids, fine ANN within the chosen shards,
    then merge. Fan-out (and driver merge) drops from W to n_shards_probe.

    Build side: cluster the corpus (e.g. distributed k-means), write one Lance
    shard + IVF index per cluster, and pass each shard's centroid here. Recall is
    traded by ``n_shards_probe`` (more probed shards = closer to exhaustive).
    """
    import numpy as np

    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    from jude.runners import _ray_shim as shim

    # coarse routing: pick the n_shards_probe shards whose centroid is nearest q
    cents = np.asarray(shard_centroids, dtype="float32")
    qv = np.asarray(query, dtype="float32")
    if metric == "cosine":
        cn = np.linalg.norm(cents, axis=1)
        cn[cn == 0] = 1.0
        cdist = 1.0 - (cents @ qv) / (cn * (np.linalg.norm(qv) or 1.0))
    elif metric in ("ip", "inner"):
        cdist = -(cents @ qv)
    else:
        diff = cents - qv
        cdist = np.einsum("ij,ij->i", diff, diff)
    npr_shards = max(1, min(n_shards_probe, len(shard_paths)))
    chosen = np.argsort(cdist)[:npr_shards].tolist()

    workers = r._ensure_workers()
    refs = [
        workers[r.mgr.worker_for(i)].vector_knn_shard.remote(
            shard_paths[s], column, list(query), k, overfetch, nprobes, metric, where)
        for i, s in enumerate(chosen)
    ]
    locals_ = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
    if not locals_:
        return pa.table({"id": pa.array([], type=pa.int64()),
                         "_distance": pa.array([], type=pa.float64())})
    merged = pa.concat_tables(locals_).combine_chunks()
    d = merged.column("_distance").to_numpy(zero_copy_only=False)
    order = np.argsort(d, kind="stable")[: int(k)]
    return merged.take(pa.array(order.tolist(), type=pa.int64()))


# --- distributed full-text (BM25) + distributed hybrid (vector + FTS) --------


def distributed_fts(
    shard_paths: list,
    column: str,
    query: str,
    *,
    k: int = 10,
    columns: Any = None,
    runner: Any = None,
) -> "pa.Table":
    """Distributed BM25 full-text search. Each shard is a Lance dataset with an
    INVERTED index on ``column``; the query fans out to all shards, each returns
    its local top-``k`` by ``_score``, and the driver merges to the global
    top-``k``. The keyword-retrieval counterpart to ``distributed_ann_knn`` —
    together they give distributed hybrid RAG.

    Correctness note: BM25 scores are computed per shard (IDF is shard-local), so
    the global ranking approximates a single-corpus BM25 — fine for RAG recall;
    exact global IDF would need a term-stat pre-pass.
    """
    r = runner
    if r is None:
        from jude.runners import get_or_create_runner

        r = get_or_create_runner()
    from jude.runners import _ray_shim as shim

    want = list(dict.fromkeys(list(columns))) if columns is not None else None
    workers = r._ensure_workers()
    refs = [
        workers[r.mgr.worker_for(i)].fts_shard.remote(path, column, str(query), k, want)
        for i, path in enumerate(shard_paths)
    ]
    locals_ = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
    if not locals_:
        return pa.table({})
    import jude

    merged = pa.concat_tables(locals_).combine_chunks()
    con = jude.connect()
    con.register("_m", merged)
    order = "_score DESC" if "_score" in merged.column_names else "1"
    return con.sql(f"SELECT * FROM _m ORDER BY {order} LIMIT {int(k)}").to_arrow()


def distributed_hybrid(
    shard_paths: list,
    text_column: str,
    vector_column: str,
    text_query: str,
    vector_query: list[float],
    *,
    k: int = 10,
    overfetch: int = 3,
    nprobes: int | None = None,
    metric: str = "cosine",
    rrf_k: int = 60,
    id_column: str = "id",
    runner: Any = None,
) -> "pa.Table":
    """Distributed HYBRID retrieval: fan out BOTH a distributed BM25 search
    (``distributed_fts``) and a distributed ANN (``distributed_ann_knn``) over the
    same shards, then fuse the two ranked lists with Reciprocal Rank Fusion (RRF)
    on the driver. The distributed counterpart of ``lance.hybrid_search`` — the
    keyword + vector recall production RAG relies on, at cluster scale.
    """
    kw = distributed_fts(shard_paths, text_column, text_query, k=k * 2, runner=runner)
    vec = distributed_ann_knn(shard_paths, vector_column, vector_query, k=k * 2,
                              overfetch=overfetch, nprobes=nprobes, metric=metric, runner=runner)

    def _keys(tbl):
        if id_column in tbl.column_names:
            return tbl.column(id_column).to_pylist()
        return list(range(tbl.num_rows))

    scores: dict = {}
    rowmap: dict = {}
    for tbl in (kw, vec):
        if tbl is None or tbl.num_rows == 0:
            continue
        keys = _keys(tbl)
        rows = tbl.to_pylist()
        for rank, (key, row) in enumerate(zip(keys, rows)):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            rowmap.setdefault(key, row)
    if not scores:
        return pa.table({})
    top = sorted(scores, key=lambda key: scores[key], reverse=True)[:k]
    fused = [dict(rowmap[key], _rrf_score=scores[key]) for key in top]
    return pa.Table.from_pylist(fused)


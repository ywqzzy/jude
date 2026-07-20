"""jude.retrieval — hybrid *analytical* retrieval: fuse vector/full-text search
with DuckDB analytics (join / aggregate / filter / window) in one query plan.

Pure retrieval (jude.vector / jude.lance) returns a top-k table. Real RAG /
analytics wants to keep going: "similar docs, but only category=x, grouped by
author, joined to a users table". That analytical step belongs in a query
optimizer — DuckDB's. This module runs the retrieval (single-node or distributed),
registers the candidates as a named relation, and hands them to ``con.sql`` so the
join/aggregate/filter/window run in DuckDB alongside the retrieval result.

This is P0 of docs/duckdb_distributed_retrieval_design.zh.md — the two-stage form
(index → candidates → SQL), correct for jude's pylance-based stack. A future
``lance_scan()`` table function would let a single SQL string do it, but requires
the Rust ``lance`` crate.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Union

import pyarrow as pa

__all__ = ["search_then_sql", "hybrid_analytical", "lance_scan", "register_search", "register_fts"]

_Candidate = Union[pa.Table, Callable[[], Any]]


def _as_table(x: Any) -> pa.Table:
    if isinstance(x, pa.Table):
        return x
    if hasattr(x, "to_arrow"):
        return x.to_arrow()
    return pa.table(x)


def search_then_sql(
    con: Any,
    sql: str,
    candidates: Mapping[str, _Candidate],
    *,
    cleanup: bool = True,
) -> pa.Table:
    """Register each retrieval result under a name, then run ``sql`` (which
    references those names) — so join/aggregate/filter/window execute in DuckDB's
    optimizer *fused with* the retrieved candidates.

    ``candidates`` maps a relation name to either an Arrow table or a zero-arg
    callable returning one (e.g. a distributed search, evaluated lazily here):

    >>> from jude import vector, retrieval
    >>> retrieval.search_then_sql(
    ...     con,
    ...     '''SELECT u.org, count(*) n, avg(hits._distance) rel
    ...        FROM hits JOIN users u ON u.id = hits.id
    ...        WHERE hits.year >= 2023 GROUP BY 1 ORDER BY rel LIMIT 20''',
    ...     candidates={"hits": lambda: vector.knn_ann_resident(path, "v", q, k=500)},
    ... )

    Returns the SQL result as an Arrow table. Registered names are dropped
    afterwards unless ``cleanup=False``.
    """
    registered: list[str] = []
    try:
        for name, src in candidates.items():
            tbl = _as_table(src() if callable(src) else src)
            con.register(name, tbl)
            registered.append(name)
        return con.sql(sql).to_arrow()
    finally:
        if cleanup:
            for name in registered:
                try:
                    con.unregister(name)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass


def hybrid_analytical(
    con: Any,
    path: str,
    sql: str,
    *,
    vector_query: list[float] | None = None,
    text_query: str | None = None,
    vector_column: str = "v",
    text_column: str = "text",
    k: int = 500,
    name: str = "hits",
    nprobes: int | None = None,
    overfetch: int = 5,
    metric: str = "cosine",
    columns: Any = None,
    runner: Any = None,
    shard_paths: list | None = None,
) -> pa.Table:
    """Convenience: retrieve top-``k`` from a Lance dataset by vector and/or
    full-text (hybrid RRF when both are given), then run ``sql`` over the
    candidates registered as ``name``. Distributed when ``shard_paths`` +
    ``runner`` are given (sharded ANN / distributed FTS / distributed hybrid),
    else single-node.

    Single-node vector retrieval uses ``knn_rerank`` so the candidates carry their
    **payload columns** (metadata), letting ``sql`` filter/group by them (pass
    ``columns`` to restrict which). The retrieval → analytics bridge for RAG: pull
    the relevant rows, then aggregate/join/filter them in SQL.
    """
    from jude import vector as _v

    if vector_query is None and text_query is None:
        raise ValueError("provide vector_query and/or text_query")

    def _retrieve() -> pa.Table:
        distributed = bool(shard_paths)
        if vector_query is not None and text_query is not None:
            if distributed:
                return _v.distributed_hybrid(shard_paths, text_column, vector_column,
                                             text_query, vector_query, k=k, overfetch=overfetch,
                                             nprobes=nprobes, metric=metric, runner=runner)
            from jude import lance as _l
            return _l.hybrid_search(path, text_column, vector_column, text_query,
                                    vector_query, k=k)
        if vector_query is not None:
            if distributed:
                return _v.distributed_ann_knn(shard_paths, vector_column, vector_query, k=k,
                                              overfetch=overfetch, nprobes=nprobes,
                                              metric=metric, runner=runner)
            # single-node: knn_rerank carries payload columns for the analytics step
            return _v.knn_rerank(path, vector_column, vector_query, k=k, overfetch=overfetch,
                                 nprobes=nprobes, metric=metric, columns=columns)
        # text only
        if distributed:
            return _v.distributed_fts(shard_paths, text_column, text_query, k=k, runner=runner)
        from jude import _lance
        return _lance.full_text_search(path, text_column, text_query, k=k)

    return search_then_sql(con, sql, {name: _retrieve})


# --- register Lance scans/searches AS DuckDB relations (one-SQL form) ---------
# These implement the design doc's P1 (lance_scan) + P2 (jude_search/jude_fts) for
# jude's pylance stack: a Lance scanner (column + predicate pushdown, and for
# search the IVF/FTS INDEX via nearest/full_text_query) is streamed as an Arrow
# RecordBatchReader that DuckDB scans lazily via con.register — so a single SQL
# string can join/aggregate/filter over a lazily-scanned or index-retrieved Lance
# relation, with the analytics fused in DuckDB's optimizer. (A pure-Rust
# lance_scan table function would need the Rust `lance` crate — a heavy, arrow-
# incompatible dep; the pylance-backed reader is the right fit and needs no fork.)


def lance_scan(
    con: Any,
    path: str,
    name: str = "lance",
    *,
    columns: Any = None,
    where: str | None = None,
    limit: int | None = None,
) -> str:
    """Register a Lance dataset as a lazily-scanned DuckDB relation ``name``.
    Column projection (``columns``) and the predicate (``where``) are pushed down
    into Lance's scanner — only the needed columns/rows are read and *streamed*
    into DuckDB (not fully materialized). Afterwards ``SELECT ... FROM {name}``
    joins/aggregates it in DuckDB. Returns ``name``.
    """
    from jude import _lance

    ds = _lance.dataset_cached(path)
    reader = ds.scanner(columns=columns, filter=where, limit=limit).to_reader()
    con.register(name, reader)
    return name


def register_search(
    con: Any,
    path: str,
    query: list[float],
    name: str = "hits",
    *,
    column: str = "v",
    k: int = 100,
    nprobes: int | None = None,
    where: str | None = None,
    columns: Any = None,
) -> str:
    """Register the top-``k`` **index-backed** vector search over a Lance dataset
    as DuckDB relation ``name`` (uses the IVF index via Lance ``nearest``; streamed,
    adds a ``_distance`` column). ``where`` is pushed down as a pre-filter (filtered
    ANN). One SQL can then retrieve-and-analyze:

    >>> retrieval.register_search(con, "docs.lance", qv, "hits", k=500, where="year>=2023")
    >>> con.sql("SELECT author, count(*) FROM hits GROUP BY author").to_arrow()
    """
    from jude import _lance

    ds = _lance.dataset_cached(path)
    nearest = {"column": column, "q": [float(x) for x in query], "k": int(k)}
    if nprobes is not None:
        nearest["nprobes"] = int(nprobes)
    kw: dict = {"columns": columns, "nearest": nearest}
    if where:
        kw["filter"] = where
        kw["prefilter"] = True
    con.register(name, ds.scanner(**kw).to_reader())
    return name


def register_fts(
    con: Any,
    path: str,
    query: str,
    name: str = "hits",
    *,
    column: str = "text",
    k: int = 100,
    columns: Any = None,
) -> str:
    """Register the top-``k`` **index-backed** BM25 full-text search over a Lance
    dataset (INVERTED index) as DuckDB relation ``name`` (streamed, adds a
    ``_score`` column), so one SQL can retrieve-and-analyze the keyword hits."""
    from jude import _lance

    ds = _lance.dataset_cached(path)
    reader = ds.scanner(columns=columns, limit=int(k),
                        full_text_query={"query": str(query), "columns": [column]}).to_reader()
    con.register(name, reader)
    return name


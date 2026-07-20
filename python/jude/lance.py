"""jude.lance — Lance dataset operations beyond basic read/write.

jude already wraps Lance basics (read/distributed-write/vector-index/scalar-
index/vector-search/versioning) as Connection methods. This module surfaces the
higher-value Lance capabilities for the LLM-data positioning that weren't
exposed yet — full-text search (hybrid RAG), point access/sampling (training
reads), incremental maintenance (upsert/add-columns/delete), and the operational
must-haves for distributed fragment writes (compaction / version cleanup).

Thin pass-throughs to jude._lance (the Rust-backed pylance layer).
"""

from __future__ import annotations

from typing import Any

from . import _lance

__all__ = [
    "create_fts_index",
    "full_text_search",
    "take",
    "sample",
    "add_columns",
    "merge_insert",
    "delete",
    "compact",
    "cleanup_old_versions",
    "create_branch",
    "list_branches",
    "checkout_branch",
    "shallow_clone",
    "create_index_uncommitted",
    "commit_index_segments",
]

# Full-text / hybrid search
create_fts_index = _lance.create_fts_index
full_text_search = _lance.full_text_search

# Point access / sampling
take = _lance.take
sample = _lance.sample

# Incremental maintenance
add_columns = _lance.add_columns
merge_insert = _lance.merge_insert
delete = _lance.delete

# Operational (post distributed-write)
compact = _lance.compact
cleanup_old_versions = _lance.cleanup_old_versions

# Branches (git-like) + distributed index build
create_branch = _lance.create_branch
list_branches = _lance.list_branches
checkout_branch = _lance.checkout_branch
shallow_clone = _lance.shallow_clone
create_index_uncommitted = _lance.create_index_uncommitted
commit_index_segments = _lance.commit_index_segments


def hybrid_search(
    path: str,
    text_column: str,
    vector_column: str,
    text_query: str,
    vector_query: list[float],
    *,
    k: int = 10,
    rrf_k: int = 60,
) -> Any:
    """Hybrid RAG retrieval on ONE Lance dataset: BM25 keyword search over
    ``text_column`` + ANN over ``vector_column``, fused by Reciprocal Rank
    Fusion (RRF). Returns the top-`k` fused rows as an Arrow table.

    Requires an FTS index on ``text_column`` (create_fts_index) and a vector
    index on ``vector_column`` (Connection.create_lance_vector_index).
    """
    import pyarrow as pa

    kw_rows = _lance.full_text_search(path, text_column, text_query, k=k * 2)
    vec_rows = _lance.vector_search(path, vector_column, vector_query, k=k * 2)

    # RRF over each ranked list, keyed by row identity (use all columns' first
    # non-vector scalar as key if present; else fall back to row position).
    def _key_list(tbl):
        # prefer an 'id' column; else the text column value
        if "id" in tbl.column_names:
            return tbl.column("id").to_pylist()
        if text_column in tbl.column_names:
            return tbl.column(text_column).to_pylist()
        return list(range(tbl.num_rows))

    scores: dict = {}
    rowmap: dict = {}
    for tbl in (kw_rows, vec_rows):
        keys = _key_list(tbl)
        rows = tbl.to_pylist()
        for rank, (kk, row) in enumerate(zip(keys, rows)):
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (rrf_k + rank + 1)
            rowmap.setdefault(kk, row)
    top = sorted(scores, key=lambda kk: scores[kk], reverse=True)[:k]
    fused = [dict(rowmap[kk], _rrf_score=scores[kk]) for kk in top]
    return pa.Table.from_pylist(fused) if fused else kw_rows.slice(0, 0)

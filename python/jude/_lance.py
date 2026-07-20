"""Lance format — read + distributed write.

Lance's writer is Rust-backed (the `lance` crate under pylance), so the data path
stays in Rust; jude orchestrates. Read scans a Lance dataset into Arrow. Write
mirrors the Iceberg pattern: single-machine `write` via `write_dataset`, and a
distributed path where each worker writes a data *fragment* and the driver
commits the fragment set as one operation (Append / Overwrite).
"""

from __future__ import annotations

import os
from typing import Any

import lance
import pyarrow as pa

# Cache opened Lance dataset handles by path so hot query loops don't reopen the
# dataset (open is not free; reopening per query dominates latency at scale).
_DS_CACHE: dict = {}

# Monotonic mutation epoch per dataset path. Every write/mutation (append,
# delete, merge_insert, add_columns, compact, optimize_indices, restore,
# index build, ...) bumps the epoch AND drops the cached handle. Downstream
# resident caches (the in-RAM vector matrix in vector.py) compare the epoch they
# loaded at against the current one and reload when it advances — so a query
# after a write never silently returns stale data. Without this, a cached handle
# / resident matrix would keep serving the pre-write snapshot forever.
_EPOCH: dict = {}


def epoch(path: str) -> int:
    """Current mutation epoch for ``path`` (0 if never mutated). Resident caches
    store the epoch they loaded at and reload when this advances."""
    return _EPOCH.get(path, 0)


def invalidate(path: str) -> None:
    """Drop the cached dataset handle for ``path`` and bump its mutation epoch.
    Call after ANY write/mutation so subsequent reads (and resident caches keyed
    on the epoch) pick up the new snapshot instead of the stale one."""
    _DS_CACHE.pop(path, None)
    _EPOCH[path] = _EPOCH.get(path, 0) + 1

# Lance keeps IVF centroids + partition postings in an in-memory LRU on the
# dataset handle. Sizing it to hold the whole index keeps ANN queries fully
# in-memory (no per-query re-read of index pages from disk). Overridable via
# env JUDE_LANCE_INDEX_CACHE (number of cache entries; big enough to cover all
# IVF partitions). 0/unset -> Lance default.
_INDEX_CACHE_SIZE = int(os.environ.get("JUDE_LANCE_INDEX_CACHE", "0")) or None


def set_index_cache_size(entries: int | None) -> None:
    """Set the in-memory Lance index-cache size (entries) used for cached dataset
    handles opened afterwards. Larger = more of the ANN index stays resident in
    RAM. Clears the handle cache so the new size takes effect on next open."""
    global _INDEX_CACHE_SIZE
    _INDEX_CACHE_SIZE = entries or None
    _DS_CACHE.clear()


def _dataset(path: str, version: Any = None):
    if version is not None:
        return lance.dataset(path, version=version)
    ds = _DS_CACHE.get(path)
    if ds is None:
        ds = (lance.dataset(path, index_cache_size=_INDEX_CACHE_SIZE)
              if _INDEX_CACHE_SIZE else lance.dataset(path))
        _DS_CACHE[path] = ds
    return ds


def dataset_cached(path: str):
    """Public: a cached Lance dataset handle (avoids per-query reopen). The handle
    also caches the ANN index in memory (see set_index_cache_size)."""
    return _dataset(path)


def read_table(path: str, columns: Any = None, filter: Any = None, version: Any = None) -> pa.Table:  # noqa: A002
    """Read a Lance dataset, optionally at a past `version` (int) or tag (str) for
    time-travel checkout; else the latest version."""
    ds = lance.dataset(path, version=version) if version is not None else lance.dataset(path)
    return ds.to_table(columns=columns, filter=filter)


def write(table: pa.Table, path: str, mode: str = "create") -> dict:
    # mode: create | append | overwrite  (Lance's own write_dataset modes)
    m = {"create": "create", "append": "append", "overwrite": "overwrite"}.get(mode, "create")
    lance.write_dataset(table, path, mode=m)
    invalidate(path)  # new snapshot — drop stale handle + bump epoch
    return {"path": path, "rows": table.num_rows}


def write_fragment(table: pa.Table, path: str) -> Any:
    """Worker side of a distributed write: write one data fragment for `path`
    (no commit) and return its (picklable) FragmentMetadata."""
    return lance.fragment.LanceFragment.create(path, table)


def commit_fragments(path: str, fragments: list, schema: pa.Schema, mode: str = "overwrite") -> dict:
    """Driver side: commit the collected fragments as one snapshot."""
    if mode == "append":
        ds = lance.dataset(path)
        op = lance.LanceOperation.Append(fragments)
        lance.LanceDataset.commit(path, op, read_version=ds.version)
    else:  # overwrite / create
        op = lance.LanceOperation.Overwrite(schema, fragments)
        lance.LanceDataset.commit(path, op)
    invalidate(path)  # committed a new snapshot
    return {"path": path, "fragments": len(fragments)}


# --- Vector search + secondary indexing (Lance's differentiators) -----------


def create_vector_index(
    path: str,
    column: str,
    index_type: str = "IVF_PQ",
    metric: str = "L2",
    num_partitions: Any = None,
    num_sub_vectors: Any = None,
    replace: bool = True,
) -> dict:
    """Build an ANN index (IVF_PQ / IVF_HNSW_SQ / …) on an embedding column so
    vector_search runs as approximate nearest-neighbour instead of a brute-force
    scan. Fills the vector-search gap neither DuckDB nor jude had natively."""
    ds = lance.dataset(path)
    kw: dict = {"index_type": index_type, "metric": metric, "replace": replace}
    if num_partitions is not None:
        kw["num_partitions"] = num_partitions
    if num_sub_vectors is not None:
        kw["num_sub_vectors"] = num_sub_vectors
    ds.create_index(column, **kw)
    invalidate(path)  # index set changed — cached handle must not mask it
    return {"path": path, "column": column, "index_type": index_type}


def create_scalar_index(path: str, column: str, index_type: str = "BTREE") -> dict:
    """Build a scalar secondary index (BTREE for ranges, BITMAP for low-cardinality
    equality) so filters on `column` skip data instead of scanning."""
    ds = lance.dataset(path)
    ds.create_scalar_index(column, index_type=index_type)
    invalidate(path)
    return {"path": path, "column": column, "index_type": index_type}


def vector_search(
    path: str,
    column: str,
    query: Any,
    k: int = 10,
    filter: Any = None,  # noqa: A002
    columns: Any = None,
    nprobes: Any = None,
    refine_factor: Any = None,
) -> pa.Table:
    """Approximate nearest-neighbour search: the `k` rows whose `column` vector is
    closest to `query`, with an added `_distance` column. `filter` pushes a
    predicate into the scan (hybrid search); `nprobes`/`refine_factor` trade
    recall for speed."""
    ds = lance.dataset(path)
    nearest: dict = {"column": column, "q": list(query), "k": int(k)}
    if nprobes is not None:
        nearest["nprobes"] = int(nprobes)
    if refine_factor is not None:
        nearest["refine_factor"] = int(refine_factor)
    return ds.to_table(nearest=nearest, filter=filter, columns=columns)


def optimize_indices(path: str) -> dict:
    """Re-globalize indices after appends: fold fragments written since the last
    index build into the index, so ANN/scalar lookups cover the whole dataset
    (a distributed write appends fragments that start out unindexed)."""
    ds = lance.dataset(path)
    ds.optimize.optimize_indices()
    invalidate(path)  # index now covers folded-in fragments
    return {"path": path, "indices": [ix.get("name") for ix in ds.list_indices()]}


def list_indices(path: str) -> list:
    return lance.dataset(path).list_indices()


# --- Git-like versioning: log / time-travel / tags / restore ----------------


def list_versions(path: str) -> pa.Table:
    """Version history (git log): one row per committed version."""
    ds = lance.dataset(path)
    rows = ds.versions()
    versions, times = [], []
    for v in rows:
        if isinstance(v, dict):
            versions.append(v.get("version"))
            times.append(str(v.get("timestamp") or v.get("created_at") or ""))
        else:
            versions.append(int(v))
            times.append("")
    return pa.table({"version": versions, "timestamp": times})


def create_tag(path: str, tag: str, version: int) -> dict:
    """Name a version (git tag)."""
    ds = lance.dataset(path)
    ds.tags.create(tag, int(version))
    return {"path": path, "tag": tag, "version": int(version)}


def list_tags(path: str) -> pa.Table:
    tags = lance.dataset(path).tags.list()
    names = list(tags.keys())
    versions = [tags[n].get("version") for n in names]
    return pa.table({"tag": names, "version": versions})


def delete_tag(path: str, tag: str) -> dict:
    lance.dataset(path).tags.delete(tag)
    return {"path": path, "tag": tag, "deleted": True}


def restore(path: str, version: Any) -> dict:
    """Roll back: make `version` (int or tag) the current/latest version — a new
    commit that restores the old state (history is preserved, git-revert style)."""
    ds = lance.dataset(path, version=version)
    ds.restore()
    invalidate(path)  # latest pointer moved
    return {"path": path, "restored_from": version, "version": ds.version}


# --- Full-text search (BM25 inverted index) — RAG hybrid search -------------


def create_fts_index(path: str, column: str, *, replace: bool = True, with_position: bool = True) -> dict:
    """Build an inverted (full-text) index on a text `column` so `full_text_search`
    runs BM25 keyword retrieval — on the SAME dataset as the vector index, giving
    hybrid (keyword + vector) RAG without a separate search engine."""
    ds = lance.dataset(path)
    ds.create_scalar_index(column, index_type="INVERTED", replace=replace, with_position=with_position)
    invalidate(path)  # FTS index added
    return {"path": path, "column": column, "index_type": "INVERTED"}


def full_text_search(path: str, column: str, query: str, k: int = 10, columns: Any = None) -> pa.Table:
    """BM25 keyword search over an FTS-indexed text `column`; top-`k` rows."""
    ds = lance.dataset(path)
    return ds.to_table(full_text_query={"query": query, "columns": [column]}, limit=int(k), columns=columns)


# --- Point access / sampling — training shuffle reads ------------------------


def take(path: str, indices: list, columns: Any = None) -> pa.Table:
    """Fetch specific rows by row index — fast columnar point access (training
    shuffle reads, id look-ups) without a full scan + filter."""
    ds = lance.dataset(path)
    return ds.take(list(indices), columns=columns)


def sample(path: str, n: int, columns: Any = None, randomize: bool = True) -> pa.Table:
    """Randomly sample `n` rows (eval subsets, quick inspection)."""
    ds = lance.dataset(path)
    return ds.sample(int(n), columns=columns, randomize_order=randomize)


# --- Incremental maintenance — clean iteration on a dataset ------------------


def add_columns(path: str, transforms: dict) -> dict:
    """Add computed columns to an existing dataset WITHOUT rewriting it
    (metadata + new column files only): {new_col: sql_expr}. e.g. backfill a
    quality score or a derived field on a TB-scale dataset cheaply."""
    ds = lance.dataset(path)
    ds.add_columns(transforms)
    invalidate(path)  # schema + data changed
    return {"path": path, "added": list(transforms.keys())}


def merge_insert(path: str, new_data: pa.Table, on: Any) -> dict:
    """Upsert: match existing rows by key column(s) `on`, update them and insert
    new ones — dedup/backfill without rewriting the whole dataset."""
    ds = lance.dataset(path)
    keys = [on] if isinstance(on, str) else list(on)
    (
        ds.merge_insert(keys)
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(new_data)
    )
    invalidate(path)  # rows upserted
    return {"path": path, "on": keys, "rows": new_data.num_rows}


def delete(path: str, predicate: str) -> dict:
    """Delete rows matching a SQL predicate (remove dirty/flagged samples)."""
    ds = lance.dataset(path)
    ds.delete(predicate)
    invalidate(path)  # rows removed — cached read must not resurrect them
    return {"path": path, "predicate": predicate}


def compact(path: str) -> dict:
    """Merge small fragments into larger ones — essential after a distributed
    write (one fragment per worker produces many small files that slow scans)."""
    ds = lance.dataset(path)
    metrics = ds.optimize.compact_files()
    invalidate(path)  # fragment layout changed
    return {"path": path, "fragments_removed": getattr(metrics, "fragments_removed", None),
            "fragments_added": getattr(metrics, "fragments_added", None)}


def cleanup_old_versions(path: str, older_than_seconds: Any = None) -> dict:
    """Reclaim storage from superseded versions (git-like history bounds disk)."""
    import datetime

    ds = lance.dataset(path)
    kw: dict = {}
    if older_than_seconds is not None:
        kw["older_than"] = datetime.timedelta(seconds=float(older_than_seconds))
    stats = ds.cleanup_old_versions(**kw)
    return {"path": path, "bytes_removed": getattr(stats, "bytes_removed", None)}


# --- branches (git-like) — Lance tags_and_branches ---------------------------


def create_branch(path: str, name: str, version: Any = None) -> dict:
    """Create a named branch at ``version`` (default: latest). Branches let you
    fork a dataset's history (A/B data configs, experiment lines) cheaply —
    git-like, no data copy."""
    ds = lance.dataset(path)
    v = version if version is not None else ds.version
    ds.create_branch(name, v)
    invalidate(path)
    return {"path": path, "branch": name, "version": v}


def list_branches(path: str) -> list:
    """List branch names of a Lance dataset."""
    b = lance.dataset(path).branches.list()
    return list(b.keys()) if isinstance(b, dict) else list(b)


def checkout_branch(path: str, name: str) -> dict:
    """Return a dataset handle at branch/version ``name`` (does not mutate the
    default)."""
    lance.dataset(path).checkout_version(name)
    return {"path": path, "checked_out": name}


def shallow_clone(path: str, target: str, ref: Any = None) -> dict:
    """Zero-copy fork of a dataset into ``target`` at ``ref`` (version/tag/branch)
    — cheap experiment branch without copying data files."""
    ds = lance.dataset(path)
    ds.shallow_clone(target, ref) if ref is not None else ds.shallow_clone(target)
    invalidate(target)  # target dataset (re)created
    return {"path": path, "clone": target, "ref": ref}


# --- distributed indexing — build index segments in parallel, then commit ----


def create_index_uncommitted(path: str, column: str, index_type: str = "IVF_PQ", **kwargs: Any) -> Any:
    """Worker side of a DISTRIBUTED index build: build an index (segment) WITHOUT
    committing it, returning the segment metadata for the driver to merge/commit.
    Multiple workers build segments over different fragments in parallel; the
    driver combines them — Lance's distributed_indexing flow. Falls back to a
    normal committed build if the installed Lance lacks the uncommitted API."""
    ds = lance.dataset(path)
    fn = getattr(ds, "create_index_uncommitted", None)
    if fn is None:
        ds.create_index(column, index_type=index_type, **kwargs)
        return {"committed": True, "path": path, "column": column}
    return fn(column, index_type=index_type, **kwargs)


def commit_index_segments(path: str, column: str, segments: list) -> dict:
    """Driver side of a distributed index build: commit the index segments built
    by the workers as one index on ``column``."""
    ds = lance.dataset(path)
    merge = getattr(ds, "commit_existing_index_segments", None) or getattr(ds, "merge_existing_index_segments", None)
    if merge is None:
        return {"committed": False, "note": "installed Lance lacks segment-commit API; use create_vector_index"}
    merge(column, segments)
    invalidate(path)
    return {"path": path, "column": column, "segments": len(segments)}

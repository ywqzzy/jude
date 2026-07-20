"""jude.runners._ray_shim — the thin Ray RPC boundary.

This module is deliberately dumb: it forwards work to Ray and collects results.
**It makes no scheduling decisions.** Every decision — how many partitions, which
worker runs task i, how large the in-flight window is, how many shuffle buckets —
is made in Rust (``jude.dist.WorkerManager``); this module only executes the plan
against Ray.

The split is enforced by review: if you are tempted to add sizing math, an env
read, or a policy branch here, it belongs in ``src/dist/worker_manager.rs``
instead. The only control flow here is the ``ray.wait``/``ray.get`` loop, which
must be Python because Ray ObjectRefs cannot cross into Rust.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import ray

if TYPE_CHECKING:
    import pyarrow as pa


def _det_hash(s: str) -> int:
    """Deterministic (cross-process) 64-bit FNV-1a hash of a string. Python's
    builtin hash() is per-process randomized, which breaks shuffle routing where
    different workers must agree on the bucket for the same key."""
    h = 0xCBF29CE484222325
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _realign(table: "pa.Table") -> "pa.Table":
    """Return a copy of ``table`` with freshly-allocated, aligned buffers.

    Ray's zero-copy Arrow deserialization hands back buffers backed by the plasma
    store, aligned to 8 bytes. DuckDB's arrow-rs C-stream importer panics when a
    buffer isn't aligned to its scalar type — and ``decimal128`` needs 16-byte
    alignment, which ``combine_chunks``/IPC do not guarantee. ``Table.take`` over
    the full index runs a C gather kernel that materializes every column into a
    fresh, type-aligned allocation, fixing decimals (and everything else) in one
    cheap pass. Empty tables are returned as-is (schema preserved)."""
    import pyarrow as pa

    if table.num_rows == 0:
        return table
    return table.take(pa.array(range(table.num_rows)))


def _decode_shard(tbl: "pa.Table", column: str, id_column: str = "id"):
    """Decode a Lance shard's id + vector column into (ids, matrix, norms) numpy
    arrays WITHOUT a per-element Python round-trip. A FixedSizeList<float32,d>
    column flattens to its contiguous child buffer, so ``.flatten().to_numpy()``
    + reshape is O(1) copies instead of materializing N*d Python floats (which is
    catastrophic at realistic dims like 768). Falls back to to_pylist only if the
    fast path is unavailable. ``id_column`` may be any type (int/str/UUID) and its
    native type is preserved; if the shard has no id column, in-shard row indices
    are used (only globally-unique when there is a single shard)."""
    import numpy as np
    import pyarrow as pa

    n = tbl.num_rows
    ids = (np.asarray(tbl.column(id_column).to_numpy(zero_copy_only=False))
           if id_column in tbl.column_names else np.arange(n))
    col = tbl.column(column).combine_chunks()
    try:
        if pa.types.is_fixed_size_list(col.type):
            d = col.type.list_size
            flat = col.flatten().to_numpy(zero_copy_only=False).astype("float32", copy=False)
            mat = flat.reshape(n, d) if n else flat.reshape(0, d)
        else:  # variable list — one alloc via numpy stack
            mat = np.asarray(col.to_pylist(), dtype="float32")
    except Exception:  # noqa: BLE001
        mat = np.asarray(col.to_pylist(), dtype="float32")
    norms = np.linalg.norm(mat, axis=1) if mat.size else np.zeros(n, dtype="float32")
    norms[norms == 0] = 1.0
    return ids, mat, norms



# ---------------------------------------------------------------------------
# Ray actor: per-partition execution on stock DuckDB (RPC-side, not scheduling)
# ---------------------------------------------------------------------------


@ray.remote
class _JudeWorker:
    """A Ray actor that executes a partition of work.

    Each actor holds its own jude Connection (stock DuckDB) so UDFs and SQL run
    close to the data. GPU actors pin CUDA_VISIBLE_DEVICES.
    """

    def __init__(self, num_gpus: int = 0):
        if num_gpus > 0:
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        import jude

        self._jude = jude
        self._conn = jude.connect()
        self._vec_cache: dict = {}  # path -> (version, ids, matrix, norms); version-stamped

    def run_sql_on_table(self, table: "pa.Table", sql_template: str) -> "pa.Table":
        """Register ``table`` as ``part`` and run ``sql_template`` against it."""
        # combine_chunks(): Ray's Arrow deserialization can hand back buffers not
        # aligned to the scalar type, which the C-stream re-import rejects.
        self._conn.register("part", _realign(table))
        try:
            return self._conn.sql(sql_template).to_arrow()
        finally:
            self._conn.unregister("part")

    def join_buckets(
        self, left: "pa.Table", right: "pa.Table", condition: str, how: str, keys: list | None = None
    ) -> "pa.Table":
        """Join two co-partitioned buckets locally (each holds all rows for a
        key range, so a local join is exact)."""
        self._conn.register("lhs", _realign(left))
        self._conn.register("rhs", _realign(right))
        try:
            join_kw = {
                "inner": "INNER JOIN",
                "left": "LEFT JOIN",
                "right": "RIGHT JOIN",
                "outer": "FULL OUTER JOIN",
                "full": "FULL OUTER JOIN",
            }.get(how.lower(), "INNER JOIN")
            # Avoid duplicate key columns in the output (SELECT * would keep both
            # lhs.k and rhs.k) by excluding the join keys from the right side.
            if keys:
                excl = ", ".join(keys)
                proj = f"lhs.*, rhs.* EXCLUDE ({excl})"
            else:
                proj = "*"
            sql = f"SELECT {proj} FROM lhs {join_kw} rhs ON {condition}"
            return self._conn.sql(sql).to_arrow()
        finally:
            self._conn.unregister("lhs")
            self._conn.unregister("rhs")

    def map_partition(self, table: "pa.Table", udf_payload: dict, batch_size: int | None) -> "pa.Table":
        """Apply a pickled UDF to a partition (in-actor, its own interpreter)."""
        import cloudpickle
        import pyarrow as pa

        fn = cloudpickle.loads(bytes.fromhex(udf_payload["fn_hex"]))
        if udf_payload.get("is_class") and isinstance(fn, type):
            fn = fn()
        batches = table.to_batches(batch_size) if batch_size else table.to_batches()
        out = []
        for b in batches:
            res = fn(pa.Table.from_batches([b]) if isinstance(b, pa.RecordBatch) else b)
            if isinstance(res, pa.RecordBatch):
                res = pa.Table.from_batches([res])
            elif isinstance(res, dict):
                res = pa.table(res)
            out.append(res)
        return pa.concat_tables(out) if out else table.slice(0, 0)

    def write_parquet_file(self, table: "pa.Table", path: str) -> dict:
        """Write one partition's Arrow table to a Parquet data file (the worker
        side of a distributed Iceberg write). Returns the path + row count so the
        driver can commit the file list. Pure I/O — no scheduling."""
        import pyarrow.parquet as pq

        pq.write_table(table, path)
        return {"path": path, "rows": table.num_rows}

    def write_lance_fragment(self, table: "pa.Table", path: str) -> Any:
        """Write one partition as a Lance data fragment (no commit) and return
        its FragmentMetadata for the driver to commit. Worker side of a
        distributed Lance write; the data path is Lance's Rust writer."""
        from jude import _lance

        return _lance.write_fragment(table, path)

    def execute_datasource_task(self, task: Any, schema: "pa.Schema") -> "pa.Table":
        """Run one DataSourceTask on this worker: drain its execute() generator
        (bounded per-chunk memory) and return the shard as one Arrow table. The
        worker side of a distributed streaming DataSource read."""
        import pyarrow as pa

        from jude.datasource import _to_batches

        batches: list = []
        for chunk in task.execute():
            for b in _to_batches(chunk, schema):
                if b.num_rows:
                    batches.append(b)
        if not batches:
            return pa.Table.from_batches([], schema=schema)
        # Use the batches' own schema (tasks may add columns beyond the declared
        # floor, e.g. a tensor `frame`).
        return pa.Table.from_batches(batches).combine_chunks()

    def vector_knn_shard(self, path: str, column: str, query: list, k: int,
                         overfetch: int, nprobes, metric: str, where=None,
                         id_column: str = "id") -> "pa.Table":
        """Local top-k of ONE pre-indexed Lance shard, for distributed sharded
        ANN over billions of vectors. Uses the shard's own IVF index (two-stage
        ANN over-fetch + exact re-rank), with an optional metadata `where`
        pre-filter pushed into the index scan. The driver merges each shard's
        local top-k into the global top-k."""
        from jude import vector as _v

        out = _v.knn_rerank(path, column, list(query), k=k, overfetch=overfetch,
                            nprobes=nprobes, metric=metric, where=where)
        # return only id + _distance — never ship the (large) vector column back
        # through Ray; the driver merges on _distance and takes ids (any id type).
        keep = [c for c in out.column_names if c in (id_column, "_distance")]
        return out.select(keep) if keep else out

    def fts_shard(self, path: str, column: str, query: str, k: int, columns) -> "pa.Table":
        """Local BM25 full-text top-k of ONE Lance shard (its own INVERTED index).
        Returns rows + their `_score`; the driver merges shard results into the
        global top-k. RPC-side execution only — routing/merge live in the driver."""
        from jude import _lance

        ds = _lance.dataset_cached(path)
        want = None
        if columns is not None:
            want = list(dict.fromkeys(list(columns)))
        out = ds.to_table(full_text_query={"query": query, "columns": [column]},
                          limit=int(k), columns=want)
        return _realign(out) if out.num_rows else out

    def _resident_shard(self, path: str, column: str, id_column: str = "id"):
        """(ids, mat, norms) for a resident Lance shard, cached on this actor and
        refreshed when the dataset's on-disk version advances. Opening the dataset
        is a cheap manifest read (no vector materialization); only a version bump
        pays the full decode. Without the version stamp a resident actor pool would
        keep scoring queries against the pre-write matrix after an append/delete."""
        import lance

        ds = lance.dataset(path)  # fresh manifest read -> current latest version
        ver = ds.version
        cached = self._vec_cache.get(path)
        if cached is None or cached[0] != ver:
            ids, mat, norms = _decode_shard(ds.to_table(columns=None), column, id_column)
            self._vec_cache[path] = (ver, ids, mat, norms)
            cached = self._vec_cache[path]
        return cached[1], cached[2], cached[3]

    def vector_exact_shard(self, path: str, column: str, query: list, k: int, metric: str,
                           id_column: str = "id") -> "pa.Table":
        """Local EXACT top-k of ONE resident Lance shard (no index). The shard's
        vectors + ids are decoded to a numpy matrix ONCE and cached on this
        (persistent) actor, so repeated queries only broadcast the query vector
        and do a numpy matmul — the data stays resident, nothing re-ships. Driver
        merges shard top-ks into the global exact top-k (100% recall)."""
        import numpy as np
        import pyarrow as pa

        ids, mat, norms = self._resident_shard(path, column, id_column)
        if mat.shape[0] == 0:
            return pa.table({"id": pa.array(ids[:0]), "_distance": pa.array([], type=pa.float64())})
        qv = np.asarray(query, dtype="float32")
        if metric == "cosine":
            qn = np.linalg.norm(qv) or 1.0
            dist = 1.0 - (mat @ qv) / (norms * qn)
        elif metric in ("ip", "inner"):
            dist = -(mat @ qv)
        else:
            diff = mat - qv
            dist = np.einsum("ij,ij->i", diff, diff)
        kk = min(k, dist.shape[0])
        part = np.argpartition(dist, kk - 1)[:kk]
        order = part[np.argsort(dist[part])]
        return pa.table({
            "id": pa.array(ids[order]),  # native id type (int/str/UUID) preserved
            "_distance": pa.array(dist[order].tolist(), type=pa.float64()),
        })

    def vector_exact_shard_batch(self, path: str, column: str, queries: list, k: int,
                                 metric: str, id_column: str = "id") -> "pa.Table":
        """BATCHED resident exact KNN: ``queries`` is a B x dim matrix. The whole
        batch is scored against the cached resident shard in a SINGLE BLAS GEMM
        (B x N distances), so RPC + decode overhead is amortized across the whole
        batch instead of paid per query — this is what makes distributed vector
        search scale near-linearly with workers (per-query fan-out is RPC-bound).
        Returns one table (qi, id, _distance) with B*k rows; the driver merges
        per-qi across workers into the global top-k."""
        import numpy as np
        import pyarrow as pa

        ids, mat, norms = self._resident_shard(path, column, id_column)
        q = np.asarray(queries, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        b = q.shape[0]
        n = mat.shape[0]
        if n == 0:
            return pa.table({"qi": pa.array([], type=pa.int32()),
                             "id": pa.array(ids[:0]),
                             "_distance": pa.array([], type=pa.float64())})
        # one big GEMM for the whole batch: B x N
        if metric == "cosine":
            qn = np.linalg.norm(q, axis=1)
            qn[qn == 0] = 1.0
            dists = 1.0 - (q @ mat.T) / (qn[:, None] * norms[None, :])
        elif metric in ("ip", "inner"):
            dists = -(q @ mat.T)
        else:  # l2sq
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            dists = (np.einsum("ij,ij->i", q, q)[:, None]
                     + np.einsum("ij,ij->i", mat, mat)[None, :]
                     - 2.0 * (q @ mat.T))
        kk = min(k, n)
        qi_out: list = []
        id_out: list = []
        d_out: list = []
        for i in range(b):
            di = dists[i]
            part = np.argpartition(di, kk - 1)[:kk]
            order = part[np.argsort(di[part])]
            qi_out.extend([i] * kk)
            id_out.extend(ids[order].tolist())
            d_out.extend(di[order].tolist())
        return pa.table({
            "qi": pa.array(qi_out, type=pa.int32()),
            "id": pa.array(id_out),  # native id type inferred (int/str/UUID)
            "_distance": pa.array(d_out, type=pa.float64()),
        })

    def kmeans_map(self, table: "pa.Table", column: str, centroids: list) -> tuple:
        """One distributed k-means map step over a shard: assign this shard's
        points to nearest centroid + accumulate per-cluster (sum, count). Returns
        (sums, counts, inertia) for the driver to merge. The Rust hot loop."""
        import numpy as np
        from jude.jude import _curate

        t = _realign(table)
        mat = np.asarray(t.column(column).to_pylist(), dtype="float32")
        n = mat.shape[0]
        dim = mat.shape[1] if n else len(centroids[0])
        if n == 0:
            k = len(centroids)
            return ([[0.0] * dim for _ in range(k)], [0] * k, 0.0)
        return _curate.kmeans_assign_accumulate(mat.reshape(-1).tolist(), n, dim, centroids)

    # --- data-curation worker RPCs (distributed jude.curate) ----------------

    def curate_map(self, table: "pa.Table", op: str, kwargs: dict) -> "pa.Table":
        """Apply a map-style (embarrassingly-parallel) curation op to one shard.
        `op` names a function in jude.curate / jude.curate_mm that takes a table
        and returns a table (chunk_text, add_content_hash, quality_filter,
        quality_signals, detect_language, language_filter, add_image_quality,
        image_quality_filter, add_image_hash). Realign for cross-process safety."""
        from jude import curate, curate_mm

        fn = getattr(curate, op, None) or getattr(curate_mm, op, None)
        if fn is None:
            raise ValueError(f"unknown curate op {op!r}")
        return _realign(fn(_realign(table), **kwargs))

    def curate_minhash_edges(self, table: "pa.Table", column: str, num_hashes: int,
                             ngram: int, bands: int, seed: int, num_buckets: int,
                             row_offset: int) -> list:
        """Producer side of distributed fuzzy dedup (recall-correct). Compute
        MinHash signatures for this shard, then route each row to a bucket for
        EACH of its LSH band keys (not just the first) — so two near-dups that
        share ANY band always co-locate in at least one bucket, matching
        single-node LSH recall. Only the lightweight (global row id, band key,
        signature) travels through the shuffle, never the full row.

        Global row id ``_rid = row_offset + local_index`` is stable across the
        whole corpus (the driver slices the table in order), so the driver can
        union-find edges across buckets and filter the original table by id.
        Returns ``num_buckets`` tables with columns (_rid, _bandkey, _sig)."""
        import pyarrow as pa
        from jude.jude import _curate

        t = _realign(table)
        texts = t.column(column).to_pylist()
        sigs = _curate.minhash_signature_batch(texts, num_hashes, ngram, seed)
        band_keys = _curate.lsh_band_keys_batch(sigs, bands)
        rid_b: list[list[int]] = [[] for _ in range(num_buckets)]
        key_b: list[list[str]] = [[] for _ in range(num_buckets)]
        sig_b: list[list] = [[] for _ in range(num_buckets)]
        for i, keys in enumerate(band_keys):
            rid = row_offset + i
            seen_bkt: set = set()  # a row lands once per bucket even if two bands collide
            ks = keys if keys else [f"_row{rid}"]  # empty text -> own singleton bucket
            for key in ks:
                bkt = _det_hash(key) % num_buckets
                if (bkt, key) in seen_bkt:
                    continue
                seen_bkt.add((bkt, key))
                rid_b[bkt].append(rid)
                key_b[bkt].append(key)
                sig_b[bkt].append(sigs[i])
        out = []
        for bkt in range(num_buckets):
            out.append(pa.table({
                "_rid": pa.array(rid_b[bkt], type=pa.int64()),
                "_bandkey": pa.array(key_b[bkt], type=pa.string()),
                "_sig": pa.array(sig_b[bkt], type=pa.list_(pa.uint64())),
            }))
        return out if num_buckets > 1 else out[0]

    def curate_fuzzy_edges_bucket(self, shard_refs: list, threshold: float) -> "pa.Table":
        """Reducer side of distributed fuzzy dedup: gather a bucket's
        (_rid, _bandkey, _sig) rows, group by exact band key (LSH candidates),
        verify each candidate pair by MinHash Jaccard >= threshold, and emit the
        surviving near-dup EDGES as (a, b) global-row-id pairs. Cross-bucket
        merging into clusters is the DRIVER's job (global union-find) — that's
        what makes the result match single-node fuzzy_dedup instead of only
        deduping within a bucket."""
        import pyarrow as pa
        from jude.jude import _curate

        shards = [t for t in ray.get(shard_refs) if t is not None and t.num_rows > 0]
        empty = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.int64())})
        if not shards:
            return empty
        merged = _realign(pa.concat_tables(shards))
        rids = merged.column("_rid").to_pylist()
        keys = merged.column("_bandkey").to_pylist()
        sigs = merged.column("_sig").to_pylist()
        # group row-indices by band key: only rows sharing a band are candidates
        by_key: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            by_key.setdefault(key, []).append(i)
        edges: set = set()
        for members in by_key.values():
            if len(members) < 2:
                continue
            # collapse identical signatures inside the group before the O(m^2)
            # verify (a template-spam band can be huge otherwise).
            leaders: dict[tuple, int] = {}
            uniq: list[int] = []
            for idx in members:
                sk = tuple(sigs[idx])
                if sk in leaders:
                    a, b = rids[leaders[sk]], rids[idx]
                    if a != b:
                        edges.add((min(a, b), max(a, b)))  # identical -> definitely dup
                else:
                    leaders[sk] = idx
                    uniq.append(idx)
            for x in range(len(uniq)):
                for y in range(x + 1, len(uniq)):
                    if _curate.signature_similarity(sigs[uniq[x]], sigs[uniq[y]]) >= threshold:
                        a, b = rids[uniq[x]], rids[uniq[y]]
                        edges.add((min(a, b), max(a, b)))
        if not edges:
            return empty
        aa = [e[0] for e in edges]
        bb = [e[1] for e in edges]
        return pa.table({"a": pa.array(aa, type=pa.int64()), "b": pa.array(bb, type=pa.int64())})

    def curate_minhash_bucketize(self, table: "pa.Table", column: str, num_hashes: int,
                                 ngram: int, bands: int, seed: int, num_buckets: int) -> list:
        """Producer side of distributed fuzzy dedup: compute MinHash signatures
        for this shard, then route each row to one of `num_buckets` output shards
        by its FIRST LSH band key (so near-dup candidates co-locate). Returns a
        list of `num_buckets` tables, each carrying the row + its signature (as a
        list<uint64>) + a stable global-ish key column `_rid` (hash of row).

        DEPRECATED for recall (first-band-only routing misses near-dups sharing a
        later band); dist_fuzzy_dedup now uses curate_minhash_edges. Kept for any
        external caller."""
        import pyarrow as pa
        from jude.jude import _curate

        t = _realign(table)
        texts = t.column(column).to_pylist()
        sigs = _curate.minhash_signature_batch(texts, num_hashes, ngram, seed)
        band_keys = _curate.lsh_band_keys_batch(sigs, bands)
        # route by first band key hash % num_buckets
        buckets: list[list[int]] = [[] for _ in range(num_buckets)]
        for i, keys in enumerate(band_keys):
            if keys:
                # deterministic hash of the band-key string (builtin hash() is
                # per-process randomized -> different workers would disagree).
                bkt = _det_hash(keys[0]) % num_buckets
            else:
                bkt = i % num_buckets
            buckets[bkt].append(i)
        out = []
        sig_col = pa.array(sigs, type=pa.list_(pa.uint64()))
        t2 = t.append_column("_minhash_sig", sig_col)
        for ids in buckets:
            out.append(t2.take(pa.array(ids, type=pa.int64())).combine_chunks())
        return out if num_buckets > 1 else out[0]

    def curate_hash_bucketize(self, table: "pa.Table", column: str, normalize: bool,
                              num_buckets: int) -> list:
        """Producer side of distributed EXACT dedup: compute content hashes and
        route rows to buckets by hash, so identical docs co-locate."""
        import pyarrow as pa
        from jude.jude import _curate

        t = _realign(table)
        hashes = _curate.content_hash_batch(t.column(column).to_pylist(), normalize)
        buckets: list[list[int]] = [[] for _ in range(num_buckets)]
        for i, h in enumerate(hashes):
            # deterministic bucketing: content_hash is hex; use its leading bits.
            # (Python's builtin hash() is per-process randomized -> WRONG here.)
            bkt = (int(h[:8], 16) % num_buckets) if h else (i % num_buckets)
            buckets[bkt].append(i)
        t2 = t.append_column("_content_hash", pa.array(hashes, type=pa.string()))
        out = [t2.take(pa.array(ids, type=pa.int64())).combine_chunks() for ids in buckets]
        return out if num_buckets > 1 else out[0]

    def curate_exact_dedup_bucket(self, shard_refs: list, keep_hash: bool) -> "pa.Table":
        """Reducer: gather a bucket's shards (all rows with hashes in this bucket)
        and drop duplicate `_content_hash`, keeping first."""
        import pyarrow as pa

        shards = [t for t in ray.get(shard_refs) if t is not None and t.num_rows > 0]
        if not shards:
            return pa.table({})
        merged = _realign(pa.concat_tables(shards))
        hashes = merged.column("_content_hash").to_pylist()
        seen: set = set()
        keep: list[int] = []
        for i, h in enumerate(hashes):
            if h is None or h in seen:
                continue
            seen.add(h)
            keep.append(i)
        out = merged.take(pa.array(keep, type=pa.int64()))
        if not keep_hash:
            out = out.drop_columns(["_content_hash"])
        return out.combine_chunks()

    def curate_fuzzy_dedup_bucket(self, shard_refs: list, threshold: float, keep_cluster: bool) -> "pa.Table":
        """Reducer: gather a bucket's shards (near-dup candidates by LSH band),
        verify candidate pairs by MinHash Jaccard >= threshold, union-find into
        clusters, keep one per cluster. Returns the surviving rows (or all rows
        annotated with a local cluster rep when keep_cluster)."""
        import pyarrow as pa
        from jude.jude import _curate

        shards = [t for t in ray.get(shard_refs) if t is not None and t.num_rows > 0]
        if not shards:
            return pa.table({})
        merged = _realign(pa.concat_tables(shards))
        sigs = merged.column("_minhash_sig").to_pylist()
        n = len(sigs)
        # Collapse EXACT-identical signatures first (cheap): a huge bucket of
        # near-identical spam otherwise blows up the O(n^2) pair scan. Group by
        # the signature tuple; only the group leaders enter the n^2 verify.
        groups: dict[tuple, int] = {}
        leader_of: list[int] = [0] * n
        leaders: list[int] = []
        for i, s in enumerate(sigs):
            key = tuple(s)
            if key in groups:
                leader_of[i] = groups[key]
            else:
                groups[key] = i
                leader_of[i] = i
                leaders.append(i)
        m = len(leaders)
        pairs: list = []
        for a in range(m):
            for b in range(a + 1, m):
                if _curate.signature_similarity(sigs[leaders[a]], sigs[leaders[b]]) >= threshold:
                    pairs.append((leaders[a], leaders[b]))
        reps = _curate.connected_components(n, pairs + [(i, leader_of[i]) for i in range(n)])
        base = merged.drop_columns(["_minhash_sig"])
        if keep_cluster:
            return base.append_column("dup_cluster", pa.array(reps, type=pa.int64())).combine_chunks()
        keep = [i for i in range(n) if reps[i] == i]
        return base.take(pa.array(keep, type=pa.int64())).combine_chunks()

    def curate_shuffle_scatter(self, table: "pa.Table", num_buckets: int, seed: int, part_id: int) -> list:
        """Producer side of a distributed GLOBAL shuffle: assign each row of this
        shard to a random output bucket (seeded per-partition for determinism),
        returning `num_buckets` sub-tables. Rows from every partition thus
        interleave across buckets — the cross-partition mixing a global shuffle
        needs. Reduce (curate_shuffle_gather) then permutes within each bucket."""
        import numpy as np
        import pyarrow as pa

        t = _realign(table)
        n = t.num_rows
        if n == 0:
            return [t for _ in range(num_buckets)] if num_buckets > 1 else t
        rng = np.random.default_rng((seed << 20) ^ part_id)
        assign = rng.integers(0, num_buckets, size=n)
        out = []
        for b in range(num_buckets):
            idx = np.nonzero(assign == b)[0]
            out.append(t.take(pa.array(idx.tolist(), type=pa.int64())).combine_chunks())
        return out if num_buckets > 1 else out[0]

    def curate_shuffle_gather(self, shard_refs: list, seed: int, bucket_id: int) -> "pa.Table":
        """Reducer side of a distributed global shuffle: concat this bucket's
        sub-shards (rows randomly routed here from every partition) and permute
        them locally, so the bucket's output is globally random."""
        import numpy as np
        import pyarrow as pa

        shards = [t for t in ray.get(shard_refs) if t is not None and t.num_rows > 0]
        if not shards:
            return pa.table({})
        merged = _realign(pa.concat_tables(shards))
        rng = np.random.default_rng((seed << 20) ^ (bucket_id + 0x9E3779B9))
        perm = rng.permutation(merged.num_rows)
        return merged.take(pa.array(perm.tolist(), type=pa.int64())).combine_chunks()

    def stream_transform(self, table, sql_template, batch_size=None):
        """Sub-batch streaming: process this partition ONE input batch at a time
        and yield each output batch as produced (a Ray streaming generator,
        called with num_returns='streaming'). Row-wise ops only (filter /
        project / scalar map) — an output batch depends only on its input batch,
        so a consumer starts on batch 0 while batch 3 is still produced; peak
        memory is O(1 batch), not O(partition)."""
        import pyarrow as pa

        bs = batch_size or 2048
        for b in table.to_batches(max_chunksize=bs):
            self._conn.register("part", pa.Table.from_batches([b]))
            try:
                yield self._conn.sql(sql_template).to_arrow()
            finally:
                self._conn.unregister("part")

    def stream_partial_agg(self, table, partial_sql, batch_size=None):
        """Streaming partial aggregation: yield a partial aggregate per input
        batch (decomposable aggregates — SUM/COUNT/MIN/MAX). The driver combines
        the partials with a final merge. `partial_sql` runs over a `part` table."""
        import pyarrow as pa

        bs = batch_size or 2048
        for b in table.to_batches(max_chunksize=bs):
            self._conn.register("part", pa.Table.from_batches([b]))
            try:
                yield self._conn.sql(partial_sql).to_arrow()
            finally:
                self._conn.unregister("part")

    def read_hive_files(self, files: list, hive_partitioning: bool = True, union_by_name: bool = False) -> "pa.Table":
        """Read a subset of a Hive-partitioned dataset's files (partition columns
        derived from the paths). Worker side of a distributed Hive read."""
        opts = f"hive_partitioning={'true' if hive_partitioning else 'false'}"
        if union_by_name:
            opts += ", union_by_name=true"
        lst = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
        return self._conn.sql(f"SELECT * FROM read_parquet([{lst}], {opts})").to_arrow()

    def read_scan(self, kind: str, spec: Any, opts: dict) -> "pa.Table":
        """Worker side of a distributed scan: read this worker's SHARD of a source
        directly (never through the driver). `spec` is a file list for
        parquet/csv/json, or (path, fragment_ids, columns, filter) for lance.
        Returns realigned Arrow so the driver can concat/register it."""
        columns = opts.get("columns")
        where = opts.get("where")
        if kind == "lance":
            import lance

            path, frag_ids = spec
            ds = lance.dataset(path)
            want = set(frag_ids)
            frags = [f for f in ds.get_fragments() if f.fragment_id in want]
            tbl = ds.scanner(fragments=frags, columns=columns, filter=where).to_table()
            return _realign(tbl) if tbl.num_rows else tbl
        lst = ", ".join("'" + f.replace("'", "''") + "'" for f in spec)
        proj = ", ".join(columns) if columns else "*"
        filt = f" WHERE {where}" if where else ""
        if kind == "parquet":
            src = f"read_parquet([{lst}])"
        elif kind == "csv":
            extra = "".join(f", {k}={v!r}" for k, v in (opts.get("csv") or {}).items())
            src = f"read_csv_auto([{lst}]{extra})"
        elif kind == "json":
            src = f"read_json_auto([{lst}])"
        else:
            raise ValueError(f"read_scan: unknown kind {kind!r}")
        out = self._conn.sql(f"SELECT {proj} FROM {src}{filt}").to_arrow()
        return _realign(out) if out.num_rows else out

    def distinct_bucket(self, shard_refs: list) -> "pa.Table":
        """Reducer side of a distributed DISTINCT: gather a bucket's shards
        (co-located duplicates) and emit distinct rows."""
        import pyarrow as pa

        shards = ray.get(shard_refs)
        merged = _realign(pa.concat_tables(shards))
        self._conn.register("_d", merged)
        try:
            return self._conn.sql("SELECT DISTINCT * FROM _d").to_arrow()
        finally:
            self._conn.unregister("_d")

    def bucketize(self, table: "pa.Table", key_expr: str, num_buckets: int) -> list:
        """Hash-partition ONE input partition into `num_buckets` buckets on the
        worker (not the driver) — the producer side of a distributed shuffle.
        Returns a list of `num_buckets` Arrow tables; call with num_returns so
        each bucket becomes its own ObjectRef that flows straight to the reducer
        (the shuffle data never lands in the driver)."""
        self._conn.register("src", _realign(table))
        try:
            out = []
            for i in range(num_buckets):
                sub = self._conn.sql(
                    f"SELECT * FROM src WHERE (hash({key_expr}) % {num_buckets}) = {i}"
                ).to_arrow()
                out.append(sub.combine_chunks())
            return out
        finally:
            self._conn.unregister("src")

    def join_bucket_group(self, left_refs: list, right_refs: list, condition: str, how: str, keys: list) -> "pa.Table":
        """Reducer side: gather this bucket's left/right shards (ObjectRefs from
        every producer, pulled from the object store — not via the driver),
        concatenate, and join locally. Co-partitioning makes the local join
        exact. Every producer returns a schema-bearing table per bucket (possibly
        0 rows), so concatenation always preserves the schema."""
        import pyarrow as pa

        lefts = ray.get(left_refs)
        rights = ray.get(right_refs)
        left_t = _realign(pa.concat_tables(lefts))
        right_t = _realign(pa.concat_tables(rights))
        return self.join_buckets(left_t, right_t, condition, how, keys)

    def sql_on_refs(self, shard_refs: list, sql_template: str) -> "pa.Table":
        """General streaming DAG exchange: gather a group of upstream shard refs
        (pulled from the object store, not via the driver), concatenate, and run
        ``sql_template`` over the result registered as ``part``. This is the
        reducer for a keyed shuffle (Aggregate/Order/Distinct final merge) and the
        per-partition local-region apply. Returns one Arrow table."""
        import pyarrow as pa

        shards = [t for t in ray.get(shard_refs) if t is not None]
        shards = [t for t in shards if t.num_rows > 0]
        if not shards:
            # Preserve schema from the first (possibly empty) shard if any.
            allrefs = ray.get(shard_refs)
            base = next((t for t in allrefs if t is not None), None)
            if base is None:
                return pa.table({})
            self._conn.register("part", base.slice(0, 0).combine_chunks())
        else:
            self._conn.register("part", _realign(pa.concat_tables(shards)))
        try:
            return self._conn.sql(sql_template).to_arrow()
        finally:
            self._conn.unregister("part")

    def setop_on_refs(self, left_refs: list, right_refs: list, kw: str) -> "pa.Table":
        """Reducer for a distributed set operation: gather this bucket's left and
        right shards, run ``(SELECT * FROM lhs) <kw> (SELECT * FROM rhs)`` where
        <kw> is UNION [ALL] / INTERSECT / EXCEPT. Co-partitioning by all columns
        makes the per-bucket set-op exact (dedup/intersect/except need matching
        rows co-located; UNION ALL is a plain concat but goes through the same
        path)."""
        import pyarrow as pa

        lefts = [t for t in ray.get(left_refs) if t is not None]
        rights = [t for t in ray.get(right_refs) if t is not None]
        lt = _realign(pa.concat_tables(lefts)) if lefts else None
        rt = _realign(pa.concat_tables(rights)) if rights else None
        if lt is None and rt is None:
            return pa.table({})
        if lt is None:
            lt = rt.slice(0, 0)
        if rt is None:
            rt = lt.slice(0, 0)
        self._conn.register("lhs", lt)
        self._conn.register("rhs", rt)
        try:
            return self._conn.sql(f"(SELECT * FROM lhs) {kw} (SELECT * FROM rhs)").to_arrow()
        finally:
            self._conn.unregister("lhs")
            self._conn.unregister("rhs")



# ---------------------------------------------------------------------------
# Cluster / actor lifecycle (pure Ray)
# ---------------------------------------------------------------------------


def ensure_init() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)


def cluster_cpus() -> int:
    return int(ray.cluster_resources().get("CPU", 1))


def cluster_gpus() -> float:
    return float(ray.cluster_resources().get("GPU", 0.0))


def cluster_nodes() -> "list[tuple[str, float, float, int]]":
    """Per-node capacities (node_id, cpu, gpu, memory_bytes) from Ray, for the
    Rust ClusterScheduler to bin-pack against. Pure read — no placement here."""
    out = []
    for n in ray.nodes():
        if not n.get("Alive", False):
            continue
        res = n.get("Resources", {})
        out.append((
            str(n.get("NodeID", "")),
            float(res.get("CPU", 0.0)),
            float(res.get("GPU", 0.0)),
            int(res.get("memory", 0)),
        ))
    return out


def make_workers(n: int, num_gpus_per_worker: int = 0) -> list:
    opts: dict[str, Any] = {}
    if num_gpus_per_worker > 0:
        opts["num_gpus"] = num_gpus_per_worker
    return [_JudeWorker.options(**opts).remote(num_gpus=num_gpus_per_worker) for _ in range(n)]


# ---------------------------------------------------------------------------
# Dispatch (pure Ray). The *window* is decided by WorkerManager.dispatch_window;
# this only runs the wait loop to that size and returns results in submission
# order.
# ---------------------------------------------------------------------------


def run_bounded(submit_fns: list[Callable[[], Any]], window: int) -> list:
    """Execute ``submit_fns`` (each returns a Ray ObjectRef) with at most
    ``window`` in flight (0 = unbounded). Results are returned in submission
    order. No policy here — ``window`` comes from the Rust WorkerManager.
    """
    if window <= 0 or window >= len(submit_fns):
        return ray.get([f() for f in submit_fns])
    results: list = [None] * len(submit_fns)
    inflight: dict = {}  # ObjectRef -> index
    next_idx = 0
    while next_idx < len(submit_fns) and len(inflight) < window:
        ref = submit_fns[next_idx]()
        inflight[ref] = next_idx
        next_idx += 1
    while inflight:
        done, _ = ray.wait(list(inflight.keys()), num_returns=1)
        ref = done[0]
        idx = inflight.pop(ref)
        results[idx] = ray.get(ref)
        if next_idx < len(submit_fns):
            nref = submit_fns[next_idx]()
            inflight[nref] = next_idx
            next_idx += 1
    return results


def get(refs: list) -> list:
    return ray.get(refs)


def stream_consume(generators: list):
    """Consume a set of Ray streaming generators (one per partition),
    round-robin, yielding each output batch as it becomes ready. The producer
    actors run concurrently, so while the driver processes one batch the others
    are already producing the next — sub-batch pipelining across partitions."""
    active = list(generators)
    while active:
        still = []
        for gen in active:
            try:
                ref = next(gen)
            except StopIteration:
                continue
            yield ray.get(ref)
            still.append(gen)
        active = still


def run_bounded_admission(
    submit_fns: "list[Callable[[], Any]]",
    resource_mgr: Any,
    per_cpu: float,
    per_gpu: float,
    per_mem: int,
    per_obj: int,
) -> list:
    """Dispatch tasks gated by a Rust ``ResourceManager``: a task launches only
    when its per-task demand can be reserved against remaining capacity, and the
    lease is released on completion. This bounds concurrency by GPUs / host
    memory / object-store bytes instead of a raw task count — the admission
    *policy* lives in Rust; this loop only forwards the reserve/launch/release
    calls to Ray. Results are returned in submission order.
    """
    n = len(submit_fns)
    results: list = [None] * n
    inflight: dict = {}  # ObjectRef -> index
    next_idx = 0

    def launch_ready() -> None:
        nonlocal next_idx
        while next_idx < n:
            if resource_mgr.try_reserve(per_cpu, per_gpu, per_mem, per_obj):
                pass
            elif not inflight:
                # Nothing in flight and can't fit — force one task so a demand
                # larger than total capacity still makes progress (runs alone).
                resource_mgr.reserve(per_cpu, per_gpu, per_mem, per_obj)
            else:
                break
            ref = submit_fns[next_idx]()
            inflight[ref] = next_idx
            next_idx += 1

    launch_ready()
    while inflight:
        done, _ = ray.wait(list(inflight.keys()), num_returns=1)
        ref = done[0]
        idx = inflight.pop(ref)
        results[idx] = ray.get(ref)
        resource_mgr.release(per_cpu, per_gpu, per_mem, per_obj)
        launch_ready()
    return results

"""jude.runners.ray — Ray-based distributed runner (partition-level orchestration).

Jude does not fork DuckDB. Instead the Ray runner orchestrates at the
*partition* level (the Daft / Ray Data model): a relation's work is split into
partitions, each partition is executed on a Ray actor (which runs stock DuckDB
plus any out-of-process UDF), and results are collected through the Ray object
store. This covers the embarrassingly-parallel scan -> map -> sink pipelines
that dominate multimodal batch inference.

**Scheduling decisions live in Rust.** This runner is a driver that delegates
every decision — partition count, partition row-slices, worker assignment,
in-flight window, shuffle bucket routing — to ``jude.dist.WorkerManager`` (the
Rust scheduling brain), and forwards the resulting plan to Ray through the thin
``jude.runners._ray_shim``. The runner itself holds no scheduling algorithm.

Ray is an optional dependency; importing this module fails cleanly if Ray is
missing, and ``jude.runners.get_or_create_runner()`` falls back to local.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import jude
from jude.runners import MaterializedResult, Runner
from jude.runners import _ray_shim as shim

if TYPE_CHECKING:
    import pyarrow as pa


def _env(name: str) -> str | None:
    import os

    return os.environ.get(f"JUDE_{name}") or os.environ.get(f"VANE_{name}")


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas not enclosed in parentheses (so f(a, b) stays one part)."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default



class RayRunner(Runner):
    name = "ray"

    def __init__(self, num_workers: int | None = None, num_gpus_per_worker: int = 0):
        shim.ensure_init()
        cluster_cpus = shim.cluster_cpus()
        self.num_workers = num_workers or max(1, cluster_cpus)
        self.num_gpus_per_worker = num_gpus_per_worker
        self._workers: list[Any] = []
        # Scheduling config (JUDE_RAY_* / VANE_RAY_* env, mirrors EnvRegistry).
        # These are kept as attributes for introspection/back-compat, but the
        # authoritative copy lives in the Rust WorkerManager below, which makes
        # every scheduling decision.
        self.size_grouping = _env_bool("RAY_SCAN_TASK_SIZE_GROUPING", True)
        self.max_task_backlog = _env_int("RAY_MAX_TASK_BACKLOG", 0)  # 0 = unlimited
        self.open_cost_bytes = _env_int("RAY_SCAN_TASK_OPEN_COST_BYTES", 4 * 1024 * 1024)
        self.min_partition_num = _env_int("RAY_SCAN_TASK_MIN_PARTITION_NUM", 0)
        # Coarse fault tolerance (the jude way — no FTE spooling/attempt machine):
        # if a distributed *read* query fails (e.g. a worker dies mid-shuffle),
        # retry the WHOLE query from scratch up to this many times. Reads are
        # functional (partition -> shuffle -> reduce, no external side effects),
        # so a clean re-run is safe. Writes are NOT auto-retried (idempotency).
        self.max_query_retries = _env_int("RAY_MAX_QUERY_RETRIES", 2)
        # The Rust scheduling brain. All decisions (partition count, row slices,
        # worker assignment, in-flight window, shuffle buckets) come from here.
        self.mgr = jude.dist.WorkerManager(
            self.num_workers,
            self.num_gpus_per_worker,
            self.size_grouping,
            self.max_task_backlog,
            self.open_cost_bytes,
            self.min_partition_num,
        )
        # Resource admission brain (GPU / host-mem / object-store aware). Capacity
        # from the Ray cluster + env overrides; decisions live in Rust. Used to
        # gate GPU work so a fleet of inference tasks never oversubscribes GPUs.
        total_gpus = float(_env_int("RAY_TOTAL_GPUS", 0)) or shim.cluster_gpus()
        self.resources = jude.dist.ResourceManager(
            0.0,
            total_gpus,
            _env_int("RAY_TOTAL_MEMORY_BYTES", 0),
            _env_int("RAY_TOTAL_OBJECT_STORE_BYTES", 0),
            _env_int("RAY_MAX_INFLIGHT", 0),
        )

    def _dispatch_admission(self, submit_fns: list, per_gpu: float) -> list:
        """Dispatch gated by the Rust ResourceManager (reserve GPU per task,
        release on completion). Falls back to count-based backpressure when there
        is no GPU demand."""
        if per_gpu <= 0:
            return self._dispatch_bounded(submit_fns)
        return shim.run_bounded_admission(submit_fns, self.resources, 0.0, per_gpu, 0, 0)

    def _ensure_workers(self) -> list[Any]:
        if not self._workers:
            self._workers = shim.make_workers(self.num_workers, self.num_gpus_per_worker)
        return self._workers

    def with_retry(self, fn, *, what: str = "query", retries: int | None = None):
        """Run a distributed read operation, retrying the WHOLE thing from
        scratch on failure (coarse fault tolerance — no FTE). On a worker fault,
        the dead actors are dropped so the retry rebuilds a fresh pool. Records
        each attempt to the observability log. Returns fn()'s result.
        """
        n = self.max_query_retries if retries is None else retries
        last_exc: BaseException | None = None
        for attempt in range(n + 1):
            try:
                return fn()
            except BaseException as e:  # noqa: BLE001 — retry any failure
                last_exc = e
                if attempt >= n:
                    break
                # a worker likely died; drop the pool so the retry rebuilds it
                self._workers = []
                try:
                    from jude import observe

                    with observe.query(f"retry {what} (attempt {attempt + 2}/{n + 1})", kind="distributed") as q:
                        q.detail(error=f"{type(e).__name__}: {e}")
                        q.done()
                except Exception:  # noqa: BLE001
                    pass
        raise last_exc

    def _target_partitions(self, table: "pa.Table") -> int:
        """Partition count from data size + worker count. Decision made in Rust
        (WorkerManager.target_partitions); this is a thin delegation kept for
        back-compat / introspection."""
        return self.mgr.target_partitions(table.nbytes, table.num_rows)

    def _partition_tables(self, relation: Any) -> list["pa.Table"]:
        table = relation.to_arrow()
        # A repartition() hint pins the count; else size-grouping decides — both
        # resolved by the Rust WorkerManager, which returns the row-slice plan.
        hint = int(getattr(relation, "num_partitions", 1) or 1)
        plan = self.mgr.partition_plan(table.num_rows, table.nbytes, hint)
        if table.num_rows == 0:
            return [table]
        # combine_chunks() so each slice has contiguous, aligned buffers — the
        # arrow-rs C-stream importer panics on the unaligned buffers that slice()
        # can leave, when the shard is re-registered on a worker.
        return [table.slice(start, length).combine_chunks() for (start, length) in plan]

    def _dispatch_bounded(self, submit_fns: list) -> list:
        """Run submit thunks with a bounded in-flight window (backpressure).

        The *window* is chosen by the Rust WorkerManager; the actual
        ray.wait/ray.get loop runs in the thin shim. Results are returned in
        submission order.
        """
        window = self.mgr.dispatch_window(len(submit_fns))
        return shim.run_bounded(submit_fns, window)

    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator["pa.Table"]:
        # Partition the materialized relation and round-robin across actors,
        # with a bounded in-flight window (backpressure).
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        submit = [
            (lambda i=i, part=part: workers[self.mgr.worker_for(i)].run_sql_on_table.remote(part, "SELECT * FROM part"))
            for i, part in enumerate(parts)
        ]
        for tbl in self._dispatch_bounded(submit):
            yield tbl

    def run_iter(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[MaterializedResult]:
        for table in self.run_iter_tables(relation, results_buffer_size):
            yield MaterializedResult(table)

    def run_write(self, relation: Any) -> dict[str, Any]:
        total = 0
        for table in self.run_iter_tables(relation):
            total += table.num_rows
        return {"rows_written": total}

    def distributed_write_iceberg(
        self, relation: Any, warehouse: str, table: str, mode: str = "append"
    ) -> str:
        """Distributed Iceberg write: partition the relation (WorkerManager
        decides how), each Ray worker writes ITS partition to a Parquet data
        file in parallel, then the driver commits the file list as one snapshot
        via the thin pyiceberg shim. jude as a distributed write engine.
        """
        import os
        import tempfile
        import uuid

        from jude import _iceberg_commit

        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        stage = os.path.join(tempfile.gettempdir(), f"jude_dist_iceberg_{uuid.uuid4().hex}")
        os.makedirs(stage, exist_ok=True)
        # Each partition -> a worker (Rust worker_for) -> writes one Parquet file.
        submit = [
            (
                lambda i=i, part=part: workers[self.mgr.worker_for(i)].write_parquet_file.remote(
                    part, os.path.join(stage, f"part-{i}.parquet")
                )
            )
            for i, part in enumerate(parts)
            if part.num_rows > 0
        ]
        results = self._dispatch_bounded(submit) if submit else []
        files = [r["path"] for r in results if r and r.get("rows", 0) > 0]
        if not files:
            # Nothing to write; still create/empty-commit an empty parquet so the
            # table exists.
            import pyarrow.parquet as pq

            empty = os.path.join(stage, "part-empty.parquet")
            pq.write_table(relation.to_arrow(), empty)
            files = [empty]
        return _iceberg_commit.commit(warehouse, table, files, mode)

    def distributed_write_lance(self, relation: Any, path: str, mode: str = "overwrite", vector_index: dict | None = None) -> dict:
        """Distributed Lance write: each Ray worker writes ITS partition as a
        Lance data fragment (Rust writer, in parallel), then the driver commits
        the fragment set as one operation (Append / Overwrite). jude as a
        distributed write engine — same shape as distributed_write_iceberg.

        `vector_index` (e.g. {"column": "emb", "num_partitions": 256,
        "num_sub_vectors": 16}) builds a GLOBAL ANN index across all committed
        fragments after the write, so vector search covers the whole distributed
        dataset (not per-fragment). For an append, call optimize_lance_indices to
        fold the new fragments into the existing global index.
        """
        from jude import _lance

        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        submit = [
            (lambda i=i, part=part: workers[self.mgr.worker_for(i)].write_lance_fragment.remote(part, path))
            for i, part in enumerate(parts)
            if part.num_rows > 0
        ]
        fragments = self._dispatch_bounded(submit) if submit else []
        fragments = [f for f in fragments if f is not None]
        schema = relation.to_arrow().schema
        if not fragments:
            # Nothing to write; create an empty dataset so the path exists.
            import pyarrow as pa

            _lance.write(pa.Table.from_batches([], schema=schema), path, mode="overwrite")
            return {"path": path, "fragments": 0}
        meta = _lance.commit_fragments(path, fragments, schema, mode=mode)
        if vector_index:
            vi = dict(vector_index)
            column = vi.pop("column")
            # Build (or, for append, extend via optimize) a global index across
            # all committed fragments.
            if mode == "append":
                try:
                    _lance.optimize_indices(path)
                except Exception:
                    _lance.create_vector_index(path, column, **vi)
            else:
                _lance.create_vector_index(path, column, **vi)
            meta["vector_index"] = column
        return meta

    def optimize_lance_indices(self, path: str) -> dict:
        """Fold fragments written since the last index build into the global
        index (call after an append so ANN/scalar lookups cover everything)."""
        from jude import _lance

        return _lance.optimize_indices(path)

    def distributed_create_vector_index(
        self, path: str, column: str, *, index_type: str = "IVF_PQ", metric: str = "cosine",
        num_partitions: int | None = None, num_sub_vectors: int | None = None, sample: int = 200_000,
    ) -> dict:
        """Distributed vector-index build. The dominant cost of an IVF index is
        training the ``num_partitions`` k-means centroids over the vectors — this
        distributes that step across workers (jude.cluster.kmeans_distributed),
        then builds the index with the precomputed centroids so the whole corpus
        isn't re-clustered single-node. Falls back to a plain single-node build
        if the installed Lance can't accept precomputed centroids; the return
        dict says which path ran.
        """
        import math

        from jude import _lance, cluster
        import numpy as np

        ds = _lance.dataset_cached(path)
        n = ds.count_rows()
        nparts = num_partitions or max(1, int(math.sqrt(n)))
        # distributed centroid training on a sample (the expensive scan, parallelized)
        samp = ds.sample(min(sample, n), columns=[column]) if hasattr(ds, "sample") else ds.to_table(columns=[column])
        centroids, _ = cluster.kmeans_distributed(samp, column, k=nparts, max_iter=15, runner=self)
        kw: dict = {"index_type": index_type, "metric": metric, "num_partitions": nparts, "replace": True}
        if num_sub_vectors is not None:
            kw["num_sub_vectors"] = num_sub_vectors
        try:
            ds.create_index(column, ivf_centroids=np.asarray(centroids, dtype="float32"), **kw)
            _lance._DS_CACHE.pop(path, None)
            return {"path": path, "column": column, "index_type": index_type,
                    "num_partitions": nparts, "centroids": "distributed-kmeans"}
        except Exception:  # noqa: BLE001 — Lance version may reject precomputed centroids
            _lance.create_vector_index(path, column, index_type=index_type, metric=metric,
                                       num_partitions=nparts, num_sub_vectors=num_sub_vectors)
            return {"path": path, "column": column, "index_type": index_type,
                    "num_partitions": nparts, "centroids": "single-node-fallback"}

    def distributed_read_hive(self, glob: str, hive_partitioning: bool = True, union_by_name: bool = False) -> "pa.Table":
        """Distributed read of a Hive-partitioned dataset: glob the leaf files,
        split them across workers (each reads its subset with partition columns
        derived from the paths), and union. File-set split decided by the Rust
        WorkerManager's partition plan; data read in parallel on the workers."""
        import glob as _glob

        import pyarrow as pa

        files = sorted(_glob.glob(glob, recursive=True))
        if not files:
            raise FileNotFoundError(f"no files match {glob!r}")
        workers = self._ensure_workers()
        n = min(len(files), max(1, self.num_workers))
        # Even split of the file list into n groups.
        step = (len(files) + n - 1) // n
        groups = [files[i : i + step] for i in range(0, len(files), step)]
        submit = [
            (lambda i=i, grp=grp: workers[self.mgr.worker_for(i)].read_hive_files.remote(grp, hive_partitioning, union_by_name))
            for i, grp in enumerate(groups)
        ]
        parts = self._dispatch_bounded(submit)
        parts = [t for t in parts if t is not None and t.num_rows > 0]
        if not parts:
            return shim.get([workers[0].read_hive_files.remote(groups[0], hive_partitioning, union_by_name)])[0]
        return pa.concat_tables(parts).combine_chunks()

    def streaming_transform(self, relation: Any, sql_template: str = "SELECT * FROM part", batch_size: int | None = None) -> "Iterator[pa.Table]":
        """Sub-batch streaming for row-wise ops (filter / project / scalar map):
        each partition's worker processes ONE input batch at a time and yields
        each output batch as produced (Ray streaming generators); the driver
        consumes them round-robin, so downstream sees batch 0 while batch N is
        still being produced. `sql_template` runs over a per-batch table `part`
        (e.g. "SELECT x, x*2 AS y FROM part WHERE x > 0"). Peak memory O(1 batch).
        """
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        gens = [
            workers[self.mgr.worker_for(i)].stream_transform.options(num_returns="streaming").remote(part, sql_template, batch_size)
            for i, part in enumerate(parts)
        ]
        yield from shim.stream_consume(gens)

    def collect(self, relation: Any) -> "pa.Table":
        """Auto-distributing executor with coarse whole-query retry. Delegates to
        _collect_once, retrying the entire distributed read on a worker fault."""
        return self.with_retry(lambda: self._collect_once(relation), what="collect")

    def _collect_once(self, relation: Any) -> "pa.Table":
        """Auto-distributing executor: inspect the query's stage DAG (from the
        Rust StagePlanner) and route to the right distributed strategy instead of
        the caller hand-picking distributed_sort / _distinct / etc.

        - ORDER BY   -> distributed_sort
        - DISTINCT   -> distributed_distinct
        - scan/map/filter/project (no shuffle) -> parallel partition scan + concat
        - Aggregate / Join -> single-node fallback (their two-phase / shuffle
          forms need the agg expressions / join keys, which the stage DAG doesn't
          carry; call distributed_aggregate / distributed_join_streaming directly).
        Returns a pyarrow Table.
        """
        import pyarrow as pa

        stages = relation.plan_stages()
        if not stages:
            return relation.to_arrow()
        # Nested shuffles (a shuffle whose input contains another shuffle) — the
        # per-op methods below can't compose these (they'd single-node the inner
        # ones), so route to the general streaming stage-DAG executor.
        try:
            if self._plan_is_nested_shuffle(relation):
                return self.execute_dag(relation)
        except Exception:  # noqa: BLE001 — fall back to the per-op router
            pass
        # Op-specific specs first — they peel Alias/Repartition to the shaping
        # operator, so `.aggregate(...).repartition(n)` still routes to the agg.
        aspec = relation.aggregate_spec()
        if aspec is not None and aspec[2]:
            from jude.runners._agg import build_two_phase

            input_rel, group, aggs = aspec
            # jude stores the aggregate list as one string; split on top-level
            # commas (respecting parens) into individual aggregate expressions.
            agg_exprs = [e for a in aggs for e in _split_top_level_commas(a)]
            partial_sql, final_sql = build_two_phase(list(group), agg_exprs)
            return self.distributed_aggregate(input_rel, partial_sql, final_sql)
        jspec = relation.join_spec()
        if jspec is not None and jspec[2] and jspec[3] in ("inner", "left", "right", "outer"):
            left, right, jkeys, how = jspec
            return self.distributed_join_streaming(left, right, list(jkeys), how)
        consumed = {i for s in stages for i in s["inputs"]}
        roots = [s for s in stages if s["id"] not in consumed] or [stages[-1]]
        root = roots[-1]
        op, keys = root["op"], root.get("partition_keys", [])
        if op == "Order":
            return self.distributed_sort(relation, keys) if keys else relation.to_arrow()
        if op == "Distinct":
            return self.distributed_distinct(relation)
        if op in ("Aggregate", "Join"):
            return relation.to_arrow()  # non-decomposable / non-equi -> single-node
        tables = [t for t in self.run_iter_tables(relation) if t.num_rows > 0]
        if not tables:
            return relation.to_arrow()
        return pa.concat_tables(tables).combine_chunks()

    def distributed_sort(self, relation: Any, keys: Any) -> "pa.Table":
        """Distributed ORDER BY: each partition sorts locally (parallel), the
        driver merges. Distribution beyond agg/join."""
        import jude
        import pyarrow as pa

        order = ", ".join(keys) if isinstance(keys, (list, tuple)) else str(keys)
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        refs = [
            workers[self.mgr.worker_for(i)].run_sql_on_table.remote(part, f"SELECT * FROM part ORDER BY {order}")
            for i, part in enumerate(parts)
        ]
        sorted_parts = [t for t in shim.get(refs) if t.num_rows > 0]
        if not sorted_parts:
            return relation.to_arrow().slice(0, 0)
        merged = pa.concat_tables(sorted_parts).combine_chunks()
        conn = jude.connect()
        conn.register("_m", merged)
        return conn.sql(f"SELECT * FROM _m ORDER BY {order}").to_arrow()

    def distributed_distinct(self, relation: Any, columns: Any = None) -> "pa.Table":
        """Distributed DISTINCT: hash-shuffle by the (all, or given) columns so
        duplicates co-locate, then local DISTINCT per bucket."""
        import pyarrow as pa

        cols = list(columns) if columns else list(relation.columns)
        key_expr = ", ".join(cols)
        b = self.mgr.shuffle_bucket_count(None)
        bucket_workers = self.mgr.shuffle_bucket_workers(None)
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        refs = [
            workers[self.mgr.worker_for(i)].bucketize.options(num_returns=b).remote(part, key_expr, b)
            for i, part in enumerate(parts)
        ]
        refs = [r if isinstance(r, list) else [r] for r in refs]
        out = [
            workers[bucket_workers[bkt]].distinct_bucket.remote([refs[p][bkt] for p in range(len(parts))])
            for bkt in range(b)
        ]
        parts_out = [t for t in shim.get(out) if t.num_rows > 0]
        if not parts_out:
            return relation.to_arrow().slice(0, 0)
        return pa.concat_tables(parts_out).combine_chunks()

    def distributed_top_k(self, relation: Any, keys: Any, k: int) -> "pa.Table":
        """Distributed ORDER BY ... LIMIT k: each partition computes its local
        top-k (parallel), the driver merges and takes the global top-k."""
        import jude
        import pyarrow as pa

        order = ", ".join(keys) if isinstance(keys, (list, tuple)) else str(keys)
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        refs = [
            workers[self.mgr.worker_for(i)].run_sql_on_table.remote(part, f"SELECT * FROM part ORDER BY {order} LIMIT {k}")
            for i, part in enumerate(parts)
        ]
        tops = [t for t in shim.get(refs) if t.num_rows > 0]
        if not tops:
            return relation.to_arrow().slice(0, 0)
        merged = pa.concat_tables(tops).combine_chunks()
        conn = jude.connect()
        conn.register("_m", merged)
        return conn.sql(f"SELECT * FROM _m ORDER BY {order} LIMIT {k}").to_arrow()

    def streaming_aggregate(self, relation: Any, partial_sql: str, final_sql: str, batch_size: int | None = None) -> "pa.Table":
        """Streaming two-phase aggregation: each partition streams a partial
        aggregate PER BATCH (bounded memory, no full-partition materialize); the
        driver unions the partials and runs `final_sql` to merge. Exact for
        decomposable aggregates (SUM/COUNT/MIN/MAX; AVG via SUM/COUNT).
        `partial_sql` runs over `part`, `final_sql` over `partials`."""
        import jude
        import pyarrow as pa

        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        gens = [
            workers[self.mgr.worker_for(i)].stream_partial_agg.options(num_returns="streaming").remote(part, partial_sql, batch_size)
            for i, part in enumerate(parts)
        ]
        partials = [t for t in shim.stream_consume(gens) if t.num_rows > 0]
        if not partials:
            return relation.to_arrow().slice(0, 0)
        merged = pa.concat_tables(partials).combine_chunks()
        conn = jude.connect()
        conn.register("partials", merged)
        return conn.sql(final_sql).to_arrow()

    def run_datasource_tasks(self, tasks: list, schema: "pa.Schema") -> list["pa.Table"]:
        """Distributed streaming DataSource read: run each DataSourceTask on a
        worker (Rust WorkerManager assigns which), each worker drains its task's
        execute() generator (bounded memory) and returns its shard. Shards flow
        back through the object store; the caller concatenates. Tasks must be
        picklable to cross into Ray workers.
        """
        workers = self._ensure_workers()
        submit = [
            (lambda i=i, task=task: workers[self.mgr.worker_for(i)].execute_datasource_task.remote(task, schema))
            for i, task in enumerate(tasks)
        ]
        return self._dispatch_bounded(submit) if submit else []

    def map_relation(self, relation: Any, udf_payload: dict, batch_size: int | None = None) -> list["pa.Table"]:
        """Distributed map_batches: apply a UDF to each partition on actors,
        with size-based partition grouping + bounded in-flight backpressure."""
        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        submit = [
            (lambda i=i, part=part: workers[self.mgr.worker_for(i)].map_partition.remote(part, udf_payload, batch_size))
            for i, part in enumerate(parts)
        ]
        # GPU work goes through resource admission (reserve/release per task);
        # CPU-only work uses the count-based bounded window.
        return self._dispatch_admission(submit, float(self.num_gpus_per_worker))

    def distributed_aggregate(
        self,
        relation: Any,
        partial_sql: str,
        final_sql: str,
    ) -> "pa.Table":
        """Two-phase distributed aggregation.

        Each partition computes ``partial_sql`` (over a table named ``part``) on
        a Ray actor; the driver unions the partials and runs ``final_sql`` (over
        a table named ``partials``) to merge. Exact for decomposable aggregates
        (SUM/COUNT/MIN/MAX, and AVG via SUM/COUNT).
        """
        import pyarrow as pa

        parts = self._partition_tables(relation)
        workers = self._ensure_workers()
        refs = [
            workers[self.mgr.worker_for(i)].run_sql_on_table.remote(part, partial_sql)
            for i, part in enumerate(parts)
        ]
        partial_tables = shim.get(refs)
        if not partial_tables:
            return relation.to_arrow().slice(0, 0)
        merged = pa.concat_tables(partial_tables)
        # combine_chunks() guarantees contiguous, aligned buffers before the
        # zero-copy C-stream re-import (concat can leave unaligned slices).
        merged = merged.combine_chunks()
        conn = jude.connect()
        conn.register("partials", merged)
        return conn.sql(final_sql).to_arrow()

    def distributed_join(
        self,
        left: Any,
        right: Any,
        keys: list[str],
        how: str = "inner",
        num_buckets: int | None = None,
    ) -> "pa.Table":
        """Distributed equi-join via hash-repartition.

        Both sides are hash-partitioned by ``keys`` into the same buckets, so
        matching keys co-locate; each bucket pair is joined on an actor and the
        results concatenated. ``keys`` are equi-join columns present in both
        sides (joined as lhs.k = rhs.k).
        """
        import pyarrow as pa

        b = self.mgr.shuffle_bucket_count(num_buckets)
        bucket_workers = self.mgr.shuffle_bucket_workers(num_buckets)
        left_t = left.to_arrow()
        right_t = right.to_arrow()

        # Hash-bucket each side by the key columns using DuckDB's hash().
        key_expr = ", ".join(keys)
        conn = jude.connect()

        def bucketize(table: "pa.Table") -> list["pa.Table"]:
            # Bucket entirely in SQL: DuckDB hashes the key and filters each
            # bucket, returning contiguous aligned Arrow (combine_chunks for the
            # zero-copy re-import on the actor side).
            conn.register("src", table)
            try:
                buckets = []
                for i in range(b):
                    sub = conn.sql(
                        f"SELECT * FROM src WHERE (hash({key_expr}) % {b}) = {i}"
                    ).to_arrow()
                    buckets.append(sub.combine_chunks())
            finally:
                conn.unregister("src")
            return buckets

        left_buckets = bucketize(left_t)
        right_buckets = bucketize(right_t)

        condition = " AND ".join(f"lhs.{k} = rhs.{k}" for k in keys)
        workers = self._ensure_workers()
        # Bucket i is joined on the worker the Rust WorkerManager assigned to it.
        refs = [
            workers[bucket_workers[i]].join_buckets.remote(
                left_buckets[i], right_buckets[i], condition, how, keys
            )
            for i in range(b)
        ]
        out = shim.get(refs)
        out = [t for t in out if t.num_rows > 0]
        if not out:
            # Return an empty table with the correct joined schema.
            return shim.get(
                [workers[bucket_workers[0]].join_buckets.remote(left_buckets[0], right_buckets[0], condition, how, keys)]
            )[0]
        return pa.concat_tables(out).combine_chunks()

    def distributed_join_streaming(
        self,
        left: Any,
        right: Any,
        keys: list[str],
        how: str = "inner",
        num_buckets: int | None = None,
    ) -> "pa.Table":
        """Distributed equi-join via a *worker-side* pipelined shuffle.

        Unlike ``distributed_join`` (which hash-buckets both sides in the driver
        — a barrier that also materializes all shuffle data on the driver), here
        each input partition is bucketized on its own worker and the per-bucket
        shards flow worker -> object store -> reducer directly. The driver only
        routes ObjectRefs (bucket -> reducer decided by the Rust WorkerManager),
        so producers (bucketize) and consumers (join) overlap and no shuffle data
        lands on the driver. jude's answer to Vane's Flight exchange, at the
        orchestration layer.
        """
        import pyarrow as pa

        b = self.mgr.shuffle_bucket_count(num_buckets)
        bucket_workers = self.mgr.shuffle_bucket_workers(num_buckets)
        workers = self._ensure_workers()
        key_expr = ", ".join(keys)

        left_parts = self._partition_tables(left)
        right_parts = self._partition_tables(right)

        # Producer stage: each partition bucketized on a worker; num_returns=b so
        # each bucket is its own ObjectRef (shuffle data stays off the driver).
        def bucketize_all(parts):
            refs = []  # refs[partition][bucket]
            for i, part in enumerate(parts):
                w = workers[self.mgr.worker_for(i)]
                refs.append(w.bucketize.options(num_returns=b).remote(part, key_expr, b))
            # Normalize: with b==1 Ray returns a single ref, not a list.
            return [r if isinstance(r, list) else [r] for r in refs]

        left_refs = bucketize_all(left_parts)
        right_refs = bucketize_all(right_parts)

        condition = " AND ".join(f"lhs.{k} = rhs.{k}" for k in keys)
        # Reducer stage: for each bucket, its shards from every producer are
        # routed (as refs) to the assigned worker, which pulls + joins them.
        refs = []
        for bkt in range(b):
            lrefs = [left_refs[p][bkt] for p in range(len(left_parts))]
            rrefs = [right_refs[p][bkt] for p in range(len(right_parts))]
            w = workers[bucket_workers[bkt]]
            refs.append(w.join_bucket_group.remote(lrefs, rrefs, condition, how, keys))
        out = [t for t in shim.get(refs) if t.num_rows > 0]
        if not out:
            return shim.get([refs[0]])[0] if refs else pa.table({})
        return pa.concat_tables(out).combine_chunks()

    # ------------------------------------------------------------------
    # General streaming stage-DAG executor (no fault tolerance).
    #
    # The methods above each distribute ONE operator; but a plan with nested
    # shuffles (e.g. aggregate -> join -> order) used to collapse everything
    # except the outermost shuffle to a single-node relation.to_arrow(). This
    # executor walks the whole DAG (via the Rust Relation.dist_step) and runs
    # EVERY shuffle boundary distributed, exchanging intermediate results as
    # lists of Ray ObjectRefs (partition shards) through the object store — never
    # gathering them to the driver between stages. Producers and consumers
    # overlap (streaming); there is deliberately no spooling/attempt machinery.
    # ------------------------------------------------------------------

    def _refs_bucketize(self, refs: list, key_expr: str, b: int, workers: list) -> list:
        """Hash-bucket each input partition ref into `b` buckets on a worker.
        Returns refs[partition][bucket]. Ray auto-dereferences each ObjectRef
        arg, so shards flow object-store -> worker without hitting the driver."""
        out = []
        for i, ref in enumerate(refs):
            w = workers[self.mgr.worker_for(i)]
            r = w.bucketize.options(num_returns=b).remote(ref, key_expr, b)
            out.append(r if isinstance(r, list) else [r])
        return out

    def _dag_partitions(self, relation: Any, depth: int = 0) -> list:
        """Execute `relation`'s stage DAG, returning its output as a list of Ray
        ObjectRefs (partition shards). Recurses on each shuffle boundary's
        children; the boundary's inputs are the children's output refs.
        """
        step = relation.dist_step()
        boundary = step["boundary"]
        local_sql = step["local_sql"]
        pushable = step["pushable"]
        workers = self._ensure_workers()

        # UDF regions or unrenderable SQL: fall back to single-node materialize
        # for this subtree (correctness over distribution).
        if step["has_udf"] or (local_sql is None and boundary == "Scan" and step.get("children")):
            import ray

            return [ray.put(relation.to_arrow())]

        # 1) Produce the boundary's output partition refs (top local NOT yet
        #    applied — a single common step below applies it).
        if boundary == "Scan":
            # Leaf (or pure partition-wise subtree): partition the source and
            # push each shard into the object store.
            parts = self._partition_tables(relation)
            raw = [
                workers[self.mgr.worker_for(i)].run_sql_on_table.remote(part, "SELECT * FROM part")
                for i, part in enumerate(parts)
            ]
        elif boundary == "Aggregate":
            from jude.runners._agg import build_two_phase

            (child,) = step["children"]
            child_refs = self._dag_partitions(child, depth + 1)
            group = list(step.get("agg_group") or [])
            aggs = step.get("agg_exprs") or []
            agg_exprs = [e for a in aggs for e in _split_top_level_commas(a)]
            # Defensive: jude keeps group keys separate from the aggregate list,
            # but if a caller folded a bare group key into the agg expr, drop it
            # (build_two_phase re-adds the group keys itself).
            gset = {g.strip() for g in group}
            agg_exprs = [e for e in agg_exprs if e.strip() not in gset]
            # Both partial and final run over the `part` placeholder (sql_on_refs
            # registers the gathered shards as `part`).
            partial_sql, final_sql = build_two_phase(group, agg_exprs, partial_table="part")
            # partial per child partition, then single-reducer final merge.
            partials = [workers[self.mgr.worker_for(i)].sql_on_refs.remote([r], partial_sql) for i, r in enumerate(child_refs)]
            raw = [workers[0].sql_on_refs.remote(partials, final_sql)]
        elif boundary == "Join" and step.get("join_keys") and step.get("how") in ("inner", "left", "right", "outer", "full"):
            left, right = step["children"]
            keys = list(step["join_keys"])
            how = step["how"]
            left_refs = self._dag_partitions(left, depth + 1)
            right_refs = self._dag_partitions(right, depth + 1)
            b = self.mgr.shuffle_bucket_count(None)
            bucket_workers = self.mgr.shuffle_bucket_workers(None)
            key_expr = ", ".join(keys)
            lb = self._refs_bucketize(left_refs, key_expr, b, workers)
            rb = self._refs_bucketize(right_refs, key_expr, b, workers)
            condition = " AND ".join(f"lhs.{k} = rhs.{k}" for k in keys)
            raw = [
                workers[bucket_workers[bkt]].join_bucket_group.remote(
                    [lb[p][bkt] for p in range(len(left_refs))],
                    [rb[p][bkt] for p in range(len(right_refs))],
                    condition, how, keys,
                )
                for bkt in range(b)
            ]
        elif boundary == "Order":
            (child,) = step["children"]
            child_refs = self._dag_partitions(child, depth + 1)
            order = ", ".join(step["keys"])
            local_sorted = [workers[self.mgr.worker_for(i)].sql_on_refs.remote([r], f"SELECT * FROM part ORDER BY {order}") for i, r in enumerate(child_refs)]
            # merge all sorted runs on one reducer (final ORDER BY).
            raw = [workers[0].sql_on_refs.remote(local_sorted, f"SELECT * FROM part ORDER BY {order}")]
        elif boundary == "Distinct":
            (child,) = step["children"]
            child_refs = self._dag_partitions(child, depth + 1)
            cols = list(child.columns)
            key_expr = ", ".join(cols) if cols else "*"
            b = self.mgr.shuffle_bucket_count(None)
            bucket_workers = self.mgr.shuffle_bucket_workers(None)
            cb = self._refs_bucketize(child_refs, key_expr, b, workers)
            raw = [
                workers[bucket_workers[bkt]].distinct_bucket.remote([cb[p][bkt] for p in range(len(child_refs))])
                for bkt in range(b)
            ]
        elif boundary == "SetOp":
            left, right = step["children"]
            kw = step["setop"]
            left_refs = self._dag_partitions(left, depth + 1)
            right_refs = self._dag_partitions(right, depth + 1)
            cols = list(left.columns)
            key_expr = ", ".join(cols) if cols else "*"
            b = self.mgr.shuffle_bucket_count(None)
            bucket_workers = self.mgr.shuffle_bucket_workers(None)
            lb = self._refs_bucketize(left_refs, key_expr, b, workers)
            rb = self._refs_bucketize(right_refs, key_expr, b, workers)
            raw = [
                workers[bucket_workers[bkt]].setop_on_refs.remote(
                    [lb[p][bkt] for p in range(len(left_refs))],
                    [rb[p][bkt] for p in range(len(right_refs))],
                    kw,
                )
                for bkt in range(b)
            ]
        elif boundary == "Repartition":
            (child,) = step["children"]
            child_refs = self._dag_partitions(child, depth + 1)
            keys = step["keys"]
            if keys:
                b = self.mgr.shuffle_bucket_count(None)
                bucket_workers = self.mgr.shuffle_bucket_workers(None)
                cb = self._refs_bucketize(child_refs, ", ".join(keys), b, workers)
                raw = [
                    workers[bucket_workers[bkt]].sql_on_refs.remote([cb[p][bkt] for p in range(len(child_refs))], "SELECT * FROM part")
                    for bkt in range(b)
                ]
            else:
                raw = child_refs
        else:
            # Unsupported boundary (non-equi join, etc.): single-node fallback.
            import ray

            return [ray.put(relation.to_arrow())]

        # 2) Apply the top partition-wise region (if any) over the boundary output.
        if local_sql:
            if pushable:
                return [workers[self.mgr.worker_for(i)].sql_on_refs.remote([r], local_sql) for i, r in enumerate(raw)]
            # Not pushable (LIMIT/SAMPLE/SUMMARIZE): gather to one, apply once.
            return [workers[0].sql_on_refs.remote(raw, local_sql)]
        return raw

    def _plan_is_nested_shuffle(self, relation: Any) -> bool:
        """True if the plan has a shuffle boundary whose children themselves
        contain a shuffle — the case the single-op methods can't handle."""
        step = relation.dist_step()
        if step["boundary"] == "Scan":
            return False
        return any(c.dist_step()["has_shuffle"] for c in step.get("children", []))

    def _describe_dag(self, relation: Any, _depth: int = 0) -> dict:
        """Walk the stage DAG to a compact description for the audit detail:
        a nested tree of {boundary, keys, children}, plus flattened
        boundary_chain (leaf→root order) and stage_names for observe stages."""
        step = relation.dist_step()
        boundary = step["boundary"]
        children = [self._describe_dag(c, _depth + 1) for c in step.get("children", [])]
        node = {
            "boundary": boundary,
            "keys": step.get("keys", []),
            "children": [c["tree"] for c in children],
        }
        # boundary_chain: distributed boundaries in dependency (leaf-first) order
        chain: list = []
        for c in children:
            chain.extend(c["boundary_chain"])
        if boundary != "Scan":
            chain.append(boundary)
        # stage_names: one per non-Scan boundary (what actually shuffles)
        return {"tree": node, "boundary_chain": chain, "stage_names": list(chain)}

    def execute_dag(self, relation: Any) -> "pa.Table":
        """Run a relation's full stage DAG distributed (general streaming
        executor) and return the assembled Arrow table. Records a detailed audit
        entry: the boundary tree, shuffle keys, partition/bucket counts, worker
        count, and the SQL of the top partition-wise region."""
        import pyarrow as pa

        from jude import observe

        step = relation.dist_step()
        boundary = step["boundary"]
        # Build a compact description of the whole DAG shape for the audit detail.
        desc = self._describe_dag(relation)
        detail = {
            "boundary": boundary,
            "shuffle_keys": step.get("keys", []),
            "join_keys": step.get("join_keys"),
            "join_how": step.get("how"),
            "setop": step.get("setop"),
            "local_sql": step.get("local_sql"),
            "num_workers": self.num_workers,
            "shuffle_buckets": self.mgr.shuffle_bucket_count(None),
            "plan_tree": desc["tree"],
            "columns": list(getattr(relation, "columns", []) or []),
        }
        chain = desc["boundary_chain"] or [boundary]
        label = f"execute_dag: {' → '.join(chain)}"
        with observe.query(label, kind="distributed") as q:
            q.detail(**detail)
            # one observe stage per distributed boundary in the DAG (dep order)
            for name in desc["stage_names"]:
                q.stage(name).done()
            refs = self._dag_partitions(relation)
            asm = q.stage("assemble", tasks_total=len(refs))
            tables = [t for t in shim.get(refs) if t is not None and t.num_rows > 0]
            asm.progress(tasks_done=len(refs), rows=sum(t.num_rows for t in tables))
            asm.done()
            if not tables:
                allt = shim.get(refs)
                base = next((t for t in allt if t is not None), None)
                out = base.slice(0, 0) if base is not None else relation.to_arrow().slice(0, 0)
                q.done(rows=out.num_rows)
                return out
            out = pa.concat_tables(tables).combine_chunks()
            q.done(rows=out.num_rows)
            return out



"""jude.datasource — pluggable, streaming, distributed-capable data ingestion.

The problem this solves (vs ``jude.sources``, which reads *all* bytes into one
Arrow table): a source can be **larger than memory**, **unbounded** (a stream),
or **user-defined** (a custom API / format). Vane's ``duckdb/datasource`` exposes
a ``DataSourceTask.execute()`` generator that yields ~10MB RecordBatches, scanned
in parallel with backpressure. jude mirrors that shape here:

    class MySource(DataSource):
        def tasks(self):                     # split the work
            for shard in my_shards:
                yield MyTask(shard)

    class MyTask(DataSourceTask):
        def execute(self):                   # a generator of RecordBatch / Table
            for chunk in read_incrementally(self.shard):
                yield chunk                  # ~O(chunk) memory, not O(dataset)

    con = jude.datasource.read(MySource())            # local streaming scan
    con = jude.datasource.read(MySource(), distributed=True)  # one task per Ray worker

Semantics:
- **Streaming**: `read_stream` yields batches one task at a time — peak memory is
  O(one task's largest batch), so datasets far bigger than RAM ingest fine.
- **Backpressure**: the consumer pulls; a slow consumer naturally stalls the
  generator (Python generator semantics), so producers don't outrun it.
- **Pluggable**: implement two tiny methods to wrap any source (custom format,
  network stream, on-the-fly synthesis).
- **Distributed**: `distributed=True` ships each task to a Ray worker (tasks must
  be picklable) and streams their output batches back through the object store.
- **Tensor-friendly**: tasks may yield fixed-shape-tensor columns (jude's tensor
  type) so multimodal frame/patch sources stream as tensors.

This is the ingestion base for scalable multimodal (streaming video frames, big
image shards) — the row-wise per-task work composes with the rest of the engine
(map_batches / SQL) unchanged.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable, Iterator

import pyarrow as pa

__all__ = [
    "DataSource",
    "DataSourceTask",
    "GeneratorSource",
    "read",
    "read_stream",
]


class DataSourceTask(abc.ABC):
    """One independently-scannable unit of a DataSource.

    ``execute()`` is a generator yielding ``pyarrow.RecordBatch`` or
    ``pyarrow.Table`` chunks. Keep each chunk bounded (~1–10MB) so peak memory
    stays O(chunk). Must be picklable if used with ``distributed=True``.
    """

    @abc.abstractmethod
    def execute(self) -> Iterator[Any]:  # -> Iterator[pa.RecordBatch | pa.Table]
        raise NotImplementedError


class DataSource(abc.ABC):
    """A splittable source of data.

    ``tasks()`` yields ``DataSourceTask`` objects (the parallel/independent
    units). ``schema()`` returns the common Arrow schema every task produces
    (required so an empty source still yields a typed, empty relation).
    """

    @abc.abstractmethod
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    @abc.abstractmethod
    def tasks(self) -> Iterable[DataSourceTask]:
        raise NotImplementedError


# --- a ready-made source wrapping plain Python generators --------------------


class _FnTask(DataSourceTask):
    """A task backed by a zero-arg callable returning an iterator of chunks."""

    def __init__(self, fn):
        self._fn = fn

    def execute(self) -> Iterator[Any]:
        yield from self._fn()


class GeneratorSource(DataSource):
    """Wrap a list of chunk-producing callables into a DataSource.

    >>> def shard0():
    ...     yield pa.record_batch({"x": [1, 2]})
    ...     yield pa.record_batch({"x": [3]})
    >>> src = GeneratorSource(schema=pa.schema([("x", pa.int64())]),
    ...                       task_fns=[shard0, shard1])

    Each callable becomes one DataSourceTask (one parallel unit). The callables
    must be picklable (module-level functions / picklable callables) to run with
    ``distributed=True``.
    """

    def __init__(self, schema: pa.Schema, task_fns: list):
        self._schema = schema
        self._task_fns = list(task_fns)

    def schema(self) -> pa.Schema:
        return self._schema

    def tasks(self) -> Iterable[DataSourceTask]:
        return [_FnTask(fn) for fn in self._task_fns]


# --- normalization helpers ---------------------------------------------------


def _to_batches(chunk: Any, schema: pa.Schema) -> list[pa.RecordBatch]:
    """Normalize a yielded chunk (RecordBatch / Table / dict) to record batches.

    ``schema`` is the source's *declared* schema, used only to reorder columns
    when the chunk carries exactly those names (so tasks may emit columns in any
    order). A chunk MAY carry extra columns beyond the declared schema (e.g. a
    tensor ``frame`` column whose extension type is awkward to declare upfront) —
    those pass through unchanged; the declared schema is a floor, not a ceiling.
    """
    if isinstance(chunk, pa.RecordBatch):
        tbl = pa.Table.from_batches([chunk])
    elif isinstance(chunk, pa.Table):
        tbl = chunk
    elif isinstance(chunk, dict):
        tbl = pa.table(chunk)
    else:
        raise TypeError(f"DataSourceTask yielded unsupported chunk type {type(chunk)!r}")
    # Reorder to the declared schema's field order only when the names match
    # exactly (a superset is left as-is).
    if tbl.schema.names != schema.names and set(tbl.schema.names) == set(schema.names):
        tbl = tbl.select(schema.names)
    return tbl.to_batches()


# --- local streaming scan ----------------------------------------------------


def read_stream(source: DataSource, *, batch_rows: int | None = None) -> Iterator[pa.RecordBatch]:
    """Stream a DataSource task-by-task, yielding RecordBatches (bounded memory).

    Peak memory is O(one task's largest emitted chunk). A slow consumer stalls
    the generator (backpressure). ``batch_rows`` optionally re-chunks output to a
    max row count.
    """
    schema = source.schema()
    emitted_any = False
    for task in source.tasks():
        for chunk in task.execute():
            for b in _to_batches(chunk, schema):
                if b.num_rows == 0:
                    continue
                emitted_any = True
                if batch_rows and b.num_rows > batch_rows:
                    tbl = pa.Table.from_batches([b])
                    for sub in tbl.to_batches(max_chunksize=batch_rows):
                        yield sub
                else:
                    yield b
    if not emitted_any:
        # Yield one empty batch so downstream keeps the schema.
        yield pa.RecordBatch.from_pylist([], schema=schema)


def read(source: DataSource, con: Any = None, *, batch_rows: int | None = None,
         distributed: bool = False, num_workers: int | None = None) -> Any:
    """Read a DataSource into a jude Relation.

    - ``distributed=False`` (default): stream locally (bounded memory) and
      register the assembled table. For sources that fit once assembled.
    - ``distributed=True``: ship each task to a Ray worker; each worker streams
      its task and returns its shard; shards are concatenated. Tasks must be
      picklable.

    For truly unbounded ingestion, iterate ``read_stream(source)`` directly and
    process batch-by-batch instead of materializing a Relation.
    """
    if con is None:
        import jude

        con = jude.connect()

    if distributed:
        table = _read_distributed(source, num_workers=num_workers)
    else:
        batches = list(read_stream(source, batch_rows=batch_rows))
        # Build from the batches' own schema (tasks may add columns beyond the
        # declared floor, e.g. a tensor `frame`); read_stream guarantees a
        # trailing empty batch carrying at least the declared schema.
        table = pa.Table.from_batches(batches) if batches else pa.Table.from_batches([], schema=source.schema())
    return con.from_arrow(table)


def _read_distributed(source: DataSource, *, num_workers: int | None = None) -> pa.Table:
    """Run each task on a Ray worker; stream its chunks; concat shards.

    Scheduling (which worker runs which task) goes through the same Rust-brained
    RayRunner used elsewhere; here we just fan tasks out and gather shards.
    ``num_workers`` is accepted for API symmetry; the runner's own configured
    worker count governs actual parallelism.
    """
    from jude.runners import get_or_create_runner

    _ = num_workers  # reserved; runner owns the worker count
    runner = get_or_create_runner()
    schema = source.schema()
    tasks = list(source.tasks())
    if not hasattr(runner, "run_datasource_tasks"):
        # No Ray runner available — fall back to local streaming.
        batches = list(read_stream(source))
        return pa.Table.from_batches(batches, schema=schema)
    shards = runner.run_datasource_tasks(tasks, schema)
    shards = [s for s in shards if s is not None and s.num_rows > 0]
    if not shards:
        return pa.Table.from_batches([], schema=schema)
    return pa.concat_tables(shards).combine_chunks()

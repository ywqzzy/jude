"""jude.pipeline._multimodal — relation-integrated multi-stage pipelines.

This is the bridge between jude's relation/SQL engine and cosmos-xenna's
multi-stage streaming pipeline model. A pipeline is a chain of independently
scaled stages (load -> decode -> transform -> write), each a cosmos ``Stage``
with its own resources/worker pool, but the *source* and *sink* are jude
relations / Arrow tables so multimodal pipelines compose with normal queries.

Data model between stages: each "sample" flowing through the pipeline is a
``pa.Table`` **shard** (a partition of rows). Shards are the unit of
parallelism — exactly like Daft/Ray Data morsels — so a stage's ``process_data``
gets a list of Arrow shards, transforms each, and returns Arrow shards. Because
shards are Arrow tables, decoders and ``map_batches`` UDFs drop straight in, and
1-to-many stages (video/document decode) just return a table with a different
row count.

Public surface:

- ``relation_to_shards`` / ``shards_to_table`` / ``shards_to_relation`` — convert
  between a jude Relation / Arrow table and a list of Arrow shards.
- ``ArrowStage`` — base cosmos Stage over Arrow shards (subclass + ``transform``).
- ``LoadFilesStage`` — read file bytes from a path column (the I/O "load" stage).
- ``DecodeStage`` — run a ``jude.multimodal`` decoder as a stage.
- ``MapBatchesStage`` — run an arbitrary batch UDF as a stage.
- ``RelationPipeline`` — fluent builder: source (Relation / Arrow / DataSource) ->
  stages -> queryable Relation. Runs on cosmos-xenna when available, else a local
  sequential engine.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pyarrow as pa

import jude.pipeline as jp

__all__ = [
    "relation_to_shards",
    "shards_to_table",
    "shards_to_relation",
    "ArrowStage",
    "LoadFilesStage",
    "DecodeStage",
    "MapBatchesStage",
    "RelationPipeline",
]


# ---------------------------------------------------------------------------
# Relation / Arrow <-> shard conversions
# ---------------------------------------------------------------------------


def _as_table(source: Any) -> pa.Table:
    """Coerce a jude Relation / Arrow table / RecordBatch to a pa.Table."""
    if isinstance(source, pa.Table):
        return source
    if isinstance(source, pa.RecordBatch):
        return pa.Table.from_batches([source])
    if hasattr(source, "to_arrow"):  # jude Relation
        return source.to_arrow()
    raise TypeError(f"cannot read a table from {type(source)!r}")


def relation_to_shards(
    source: Any,
    *,
    num_shards: int | None = None,
    rows_per_shard: int | None = None,
) -> list[pa.Table]:
    """Split a jude Relation / Arrow table into a list of Arrow shards.

    With neither argument, defaults to ``rows_per_shard=32``. Empty inputs yield
    a single empty shard so the schema still propagates through the pipeline.
    """
    table = _as_table(source)
    n = table.num_rows
    if n == 0:
        return [table]
    if num_shards is not None and num_shards > 0:
        size = max(1, -(-n // num_shards))  # ceil division
    else:
        size = rows_per_shard if (rows_per_shard and rows_per_shard > 0) else 32
    shards = []
    for start in range(0, n, size):
        shards.append(table.slice(start, min(size, n - start)))
    return shards or [table]


def shards_to_table(shards: Sequence[pa.Table]) -> pa.Table:
    """Concatenate Arrow shards back into a single aligned pa.Table."""
    tables = [s for s in shards if isinstance(s, pa.Table) and s.num_rows > 0]
    if not tables:
        # keep schema from the first shard if present
        for s in shards:
            if isinstance(s, pa.Table):
                return s
        return pa.table({})
    combined = pa.concat_tables(tables, promote_options="default")
    # arrow-rs C-stream importer panics on unaligned buffers; normalize first.
    return combined.combine_chunks()


def shards_to_relation(con: Any, shards: Sequence[pa.Table]) -> Any:
    """Land Arrow shards as a queryable jude Relation on ``con``."""
    if con is None:
        import jude

        con = jude.connect()
    return con.from_arrow(shards_to_table(shards))


# ---------------------------------------------------------------------------
# Stages (cosmos-xenna Stage over Arrow shards)
# ---------------------------------------------------------------------------


class ArrowStage(jp.Stage):
    """Base pipeline stage operating on Arrow shards.

    Subclass and implement ``transform(table) -> table`` (1:1, 1:many, or
    filtering). ``process_data`` receives a list of shards, applies ``transform``
    to each, and drops ``None`` / empty results. Resources default to
    ``cpus``/``gpus`` set at construction so each stage can be scaled
    independently by cosmos.
    """

    def __init__(self, *, cpus: float = 1.0, gpus: float = 0.0, batch_size: int = 1):
        self._cpus = cpus
        self._gpus = gpus
        self._batch_size = batch_size

    @property
    def required_resources(self) -> Any:
        return jp.Resources(cpus=self._cpus, gpus=self._gpus)

    @property
    def stage_batch_size(self) -> int:
        return self._batch_size

    def transform(self, table: pa.Table) -> pa.Table | None:
        raise NotImplementedError

    def process_data(self, samples: list) -> list:
        out: list[pa.Table] = []
        for shard in samples:
            if not isinstance(shard, pa.Table):
                shard = _as_table(shard)
            res = self.transform(shard)
            if res is None:
                continue
            if not isinstance(res, pa.Table):
                raise TypeError(f"{type(self).__name__}.transform must return a pa.Table, got {type(res)!r}")
            if res.num_rows > 0:
                out.append(res)
        return out


class LoadFilesStage(ArrowStage):
    """The I/O "load" stage: read file bytes for each row's path column.

    Turns a shard of ``path`` rows into a shard with an added binary
    ``out_column`` (the encoded bytes) — the download step of a multimodal
    pipeline, expressed as its own independently scaled stage.
    """

    def __init__(self, *, path_column: str = "path", out_column: str = "data", cpus: float = 1.0, batch_size: int = 1):
        super().__init__(cpus=cpus, batch_size=batch_size)
        self.path_column = path_column
        self.out_column = out_column

    def transform(self, table: pa.Table) -> pa.Table:
        blobs = []
        for path in table.column(self.path_column).to_pylist():
            with open(path, "rb") as fh:
                blobs.append(fh.read())
        return table.append_column(self.out_column, pa.array(blobs, type=pa.binary()))


# decoder kind -> (callable, default kwargs)
def _decoder_for(kind: str) -> Callable:
    from jude.multimodal import (
        decode_audio_batch,
        decode_document_batch,
        decode_image_batch,
        decode_video_batch,
    )

    table = {
        "image": decode_image_batch,
        "audio": decode_audio_batch,
        "video": decode_video_batch,
        "document": decode_document_batch,
    }
    if kind not in table:
        raise ValueError(f"unknown decoder kind {kind!r}; expected one of {sorted(table)}")
    return table[kind]


class DecodeStage(ArrowStage):
    """Run a ``jude.multimodal`` decoder (image/audio/video/document) as a stage."""

    def __init__(self, kind: str, *, cpus: float = 1.0, gpus: float = 0.0, batch_size: int = 1, **decoder_kwargs: Any):
        super().__init__(cpus=cpus, gpus=gpus, batch_size=batch_size)
        self.kind = kind
        self.decoder_kwargs = decoder_kwargs

    def transform(self, table: pa.Table) -> pa.Table:
        decoder = _decoder_for(self.kind)
        return decoder(table, **self.decoder_kwargs)


class MapBatchesStage(ArrowStage):
    """Run an arbitrary batch UDF ``(pa.Table) -> pa.Table`` as a stage."""

    def __init__(self, fn: Callable[[pa.Table], pa.Table], *, cpus: float = 1.0, gpus: float = 0.0, batch_size: int = 1):
        super().__init__(cpus=cpus, gpus=gpus, batch_size=batch_size)
        self.fn = fn

    def transform(self, table: pa.Table) -> pa.Table:
        from jude.execution._common import coerce_table

        return coerce_table(self.fn(table))


# ---------------------------------------------------------------------------
# RelationPipeline — fluent builder with relation source + sink
# ---------------------------------------------------------------------------


def _pickle_by_value(stage: Any) -> None:
    """Best-effort: register the module defining a stage's user callable by value
    so cosmos/Ray workers don't need to import a test/local module (same gotcha
    as jude.execution.serialize_udf)."""
    import importlib

    import cloudpickle

    targets = [type(stage)]
    fn = getattr(stage, "fn", None)
    if fn is not None:
        targets.append(fn)
    for obj in targets:
        mod = getattr(obj, "__module__", None)
        if mod and mod not in ("builtins", "__main__"):
            try:
                cloudpickle.register_pickle_by_value(importlib.import_module(mod))
            except Exception:
                pass


class RelationPipeline:
    """Fluent multi-stage pipeline: relation/source in -> stages -> relation out.

    Build a chain of stages, then materialize with :meth:`to_relation` (queryable
    jude Relation) or :meth:`run` (Arrow table). The chain runs on cosmos-xenna
    when ``engine='cosmos'`` (or ``'auto'`` and cosmos is installed), else on a
    local sequential engine (fast, order-preserving) that shares the exact same
    ``Stage`` API — so the pipeline you develop locally runs unchanged on cosmos.
    """

    def __init__(
        self,
        *,
        num_shards: int | None = None,
        rows_per_shard: int | None = None,
        engine: str = "local",
        pipeline_config: Any = None,
    ):
        self._input_table: pa.Table | None = None
        # A thunk returning a FRESH iterator of input shards, for a streaming
        # source (from_datasource). When set, the local engine pulls one input
        # shard at a time through all stages (depth-first) — bounded input memory,
        # never materializing the whole source into one table.
        self._input_shard_iter: Any = None
        self._read_bytes_source: Any = None  # a DataSource whose bytes are loaded by a stage
        self._stages: list[Any] = []
        self.num_shards = num_shards
        self.rows_per_shard = rows_per_shard
        self.engine = engine
        # Optional cosmos PipelineConfig override (e.g. monitoring_verbosity_level
        # for observability). None -> jude's default BATCH config.
        self.pipeline_config = pipeline_config

    # ---- sources ----

    @classmethod
    def from_relation(cls, rel: Any, **kw: Any) -> "RelationPipeline":
        p = cls(**kw)
        p._input_table = _as_table(rel)
        return p

    @classmethod
    def from_table(cls, table: pa.Table, **kw: Any) -> "RelationPipeline":
        p = cls(**kw)
        p._input_table = _as_table(table)
        return p

    @classmethod
    def from_source(cls, source: Any, *, read_bytes: bool = False, **kw: Any) -> "RelationPipeline":
        """Start from a ``jude.sources`` DataSource.

        With ``read_bytes=False`` (default) the pipeline starts from file
        *metadata* (path + size) and you add a :meth:`load_files` stage to read
        bytes — so loading is its own scalable stage. With ``read_bytes=True``
        the bytes are read eagerly into the input table.
        """
        p = cls(**kw)
        p._input_table = source.to_arrow(read_bytes=read_bytes)
        return p

    @classmethod
    def from_datasource(cls, source: Any, *, batch_rows: int | None = None, **kw: Any) -> "RelationPipeline":
        """Start from a streaming ``jude.datasource.DataSource``.

        Unlike :meth:`from_source` (which eagerly reads all bytes into one
        table), this consumes the source LAZILY: the local engine pulls one
        input shard (a ``read_stream`` batch of ``batch_rows`` rows) at a time and
        runs it through the whole stage chain before pulling the next — so the
        source is never materialized into a single table and input memory is
        bounded to one shard (see :meth:`run_streaming` for a fully-lazy sink).
        A larger-than-memory source therefore feeds the pipeline without OOM.
        """
        p = cls(**kw)
        # store a thunk, not a materialized list: run()/run_streaming() call it to
        # get a fresh iterator each execution (the stream is re-readable per run).
        def _iter():
            import pyarrow as pa

            from jude import datasource
            for b in datasource.read_stream(source, batch_rows=batch_rows):
                yield pa.Table.from_batches([b])
        p._input_shard_iter = _iter
        p._stream_schema = source.schema()
        return p

    # ---- stages ----

    def add_stage(self, stage: Any) -> "RelationPipeline":
        self._stages.append(stage)
        return self

    def load_files(self, *, path_column: str = "path", out_column: str = "data", cpus: float = 1.0) -> "RelationPipeline":
        return self.add_stage(LoadFilesStage(path_column=path_column, out_column=out_column, cpus=cpus))

    def decode(self, kind: str, *, cpus: float = 1.0, gpus: float = 0.0, **decoder_kwargs: Any) -> "RelationPipeline":
        return self.add_stage(DecodeStage(kind, cpus=cpus, gpus=gpus, **decoder_kwargs))

    def map_batches(self, fn: Callable[[pa.Table], pa.Table], *, cpus: float = 1.0, gpus: float = 0.0) -> "RelationPipeline":
        return self.add_stage(MapBatchesStage(fn, cpus=cpus, gpus=gpus))

    # ---- LLM data-curation stages (jude.curate) ----

    def chunk(self, *, cpus: float = 1.0, **kwargs: Any) -> "RelationPipeline":
        """Add a text-chunking stage (C5): 1 row -> many chunk rows."""
        from jude import curate

        return self.add_stage(curate.ChunkStage(cpus=cpus, **kwargs))

    def quality_filter(self, *, cpus: float = 1.0, **kwargs: Any) -> "RelationPipeline":
        """Add a quality-filter stage (C3): drop/annotate low-quality rows."""
        from jude import curate

        return self.add_stage(curate.QualityFilterStage(cpus=cpus, **kwargs))

    def content_hash(self, *, cpus: float = 1.0, **kwargs: Any) -> "RelationPipeline":
        """Add a content-hash stage (C2): the exact-dedup key column."""
        from jude import curate

        return self.add_stage(curate.ContentHashStage(cpus=cpus, **kwargs))

    # ---- execution ----

    def _shards(self) -> list[pa.Table]:
        if self._input_shard_iter is not None:
            return list(self._input_shard_iter())  # materialize the stream (breadth-first fallback)
        if self._input_table is None:
            raise ValueError("RelationPipeline has no source; use from_relation/from_table/from_source/from_datasource")
        return relation_to_shards(self._input_table, num_shards=self.num_shards, rows_per_shard=self.rows_per_shard)

    def _use_cosmos(self) -> bool:
        if self.engine == "cosmos":
            if not jp.is_cosmos_backed():
                st = jp.cosmos_status()
                if st["kind"] == "import-failed":
                    raise RuntimeError(
                        "engine='cosmos' but cosmos-xenna failed to import "
                        f"(likely a version skew): {st['error']}")
                raise RuntimeError("engine='cosmos' but cosmos-xenna is not installed")
            return True
        if self.engine == "auto":
            return jp.is_cosmos_backed()
        return False

    def run(self) -> pa.Table:
        """Execute the pipeline and return the sink as an Arrow table.

        Records into jude's observability registry (jude.observe) so multi-stage
        pipelines show up on the dashboard alongside SQL/UDF work: one query per
        run, and a REAL per-stage funnel (rows in→out per stage) on the local
        engine.
        """
        from jude import observe

        use_cosmos = self._use_cosmos()
        engine = "cosmos" if use_cosmos else "local"
        # streaming source + local engine: pull one input shard at a time through
        # all stages (bounded input memory), never materializing the source.
        streaming = self._input_shard_iter is not None and not use_cosmos
        shards = None if streaming else self._shards()
        if not self._stages:
            return shards_to_table(shards if shards is not None else self._shards())
        with observe.query(f"pipeline[{engine}]: {len(self._stages)} stages", kind="pipeline") as q:
            if use_cosmos:
                # cosmos owns execution + returns only the last stage's output, so
                # per-stage row counts aren't observable here; record names only.
                for st in self._stages:
                    q.stage(type(st).__name__).done()
                out = self._run_cosmos(shards)
            elif streaming:
                out = list(self._iter_local_streaming(self._input_shard_iter(), q))
            else:
                out = self._run_local(shards, q)  # records the real per-stage funnel
            table = shards_to_table(out)
            q.done(rows=table.num_rows, nbytes=table.nbytes)
            return table

    def run_streaming(self, *, con: Any = None) -> "Iterator[pa.Table]":
        """Fully-lazy execution: yield output shards one at a time as each input
        shard is pulled through the whole stage chain. Bounded memory end-to-end
        (input AND output) — the caller consumes/writes each shard without ever
        holding the whole result. Local engine only; requires a streaming source
        (from_datasource) or falls back to iterating the materialized shards.

        Note: stages here are per-shard maps/filters/explodes (the ArrowStage
        contract), so depth-first streaming is exact; a global-shuffle stage
        (cross-shard dedup) would need the materialized ``run()`` path instead.
        """
        src = self._input_shard_iter() if self._input_shard_iter is not None else iter(self._shards())
        if not self._stages:
            yield from src
            return
        yield from self._iter_local_streaming(src, None)

    def write_streaming(self, path: str, *, fmt: str = "lance", **storage_options: Any) -> dict:
        """Execute the pipeline and WRITE each output shard as it is produced —
        bounded memory end to end (L0.3): the source is streamed in shard by
        shard, each output shard is written out and freed, so neither the whole
        input nor the whole output is ever held. ``fmt="lance"`` appends each
        shard to one Lance dataset at ``path``; ``fmt="parquet"`` writes numbered
        parquet shards (``path/part-00000.parquet`` …) to any fsspec URL
        (local / s3:// MinIO). Returns a manifest (shards written, rows)."""
        shards = 0
        rows = 0
        if fmt == "lance":
            from jude import _lance

            for i, shard in enumerate(self.run_streaming()):
                if shard.num_rows == 0:
                    continue
                _lance.write(shard, path, mode="create" if shards == 0 else "append")
                shards += 1
                rows += shard.num_rows
        elif fmt == "parquet":
            from jude import storage

            for i, shard in enumerate(self.run_streaming()):
                if shard.num_rows == 0:
                    continue
                storage.write_parquet(shard, f"{path.rstrip('/')}/part-{i:05d}.parquet",
                                     **storage_options)
                shards += 1
                rows += shard.num_rows
        else:
            raise ValueError(f"unknown fmt {fmt!r}; use 'lance' or 'parquet'")
        return {"path": path, "format": fmt, "shards": shards, "rows": rows}

    def to_relation(self, con: Any = None) -> Any:
        """Execute the pipeline and land the sink as a queryable jude Relation."""
        if con is None:
            import jude

            con = jude.connect()
        return con.from_arrow(self.run())

    def _run_local(self, shards: list[pa.Table], q: Any = None) -> list[pa.Table]:
        """Sequential in-process engine sharing the cosmos Stage API. When an
        observe query handle ``q`` is given, record a real funnel: rows in→out
        (and dropped) per stage, so the dashboard shows which stage filtered how
        much."""
        data = shards
        for stage in self._stages:
            stage.setup(None)
        try:
            for stage in self._stages:
                sh = q.stage(type(stage).__name__) if q is not None else None
                bs = max(1, int(getattr(stage, "stage_batch_size", 1)))
                out: list[pa.Table] = []
                for start in range(0, len(data), bs):
                    batch = data[start : start + bs]
                    res = stage.process_data(batch)
                    if res:
                        out.extend(res)
                data = out
                if sh is not None:
                    rows_out = sum(t.num_rows for t in data)
                    nbytes = sum(t.nbytes for t in data)
                    # real funnel: rows_in→rows_out (dropped = rows_in - rows_out)
                    # recorded on the stage; the dashboard StagesPanel renders it.
                    sh.progress(rows=rows_out, nbytes=nbytes)
                    sh.done()
        finally:
            for stage in self._stages:
                try:
                    stage.destroy()
                except Exception:
                    pass
        return data

    def _iter_local_streaming(self, shard_iter: Any, q: Any = None) -> "Iterator[pa.Table]":
        """Depth-first local engine: run EACH input shard through the whole stage
        chain before pulling the next, yielding the final stage's output shards.
        Bounded input memory (one shard in flight, not the whole source). Records
        the same real per-stage funnel as _run_local (aggregated over the stream).
        """
        stages = self._stages
        for stage in stages:
            stage.setup(None)
        handles = [q.stage(type(s).__name__) if q is not None else None for s in stages]
        rows_acc = [0] * len(stages)
        bytes_acc = [0] * len(stages)
        try:
            for in_shard in shard_iter:
                current = [in_shard]
                for i, stage in enumerate(stages):
                    bs = max(1, int(getattr(stage, "stage_batch_size", 1)))
                    nxt: list[pa.Table] = []
                    for start in range(0, len(current), bs):
                        res = stage.process_data(current[start : start + bs])
                        if res:
                            nxt.extend(res)
                    current = nxt
                    for t in current:
                        rows_acc[i] += t.num_rows
                        bytes_acc[i] += t.nbytes
                    if not current:
                        break  # this input shard yielded nothing downstream
                for t in current:
                    yield t
            for i, sh in enumerate(handles):
                if sh is not None:
                    sh.progress(rows=rows_acc[i], nbytes=bytes_acc[i])
                    sh.done()
        finally:
            for stage in stages:
                try:
                    stage.destroy()
                except Exception:
                    pass

    def _run_cosmos(self, shards: list[pa.Table]) -> list[pa.Table]:
        """Run on cosmos-xenna: each stage a StageSpec, shards as input_data."""
        for stage in self._stages:
            _pickle_by_value(stage)
        # A user-supplied PipelineConfig wins; otherwise jude's default enables
        # cosmos-xenna's built-in FAULT TOLERANCE (X.1): retry failed stage tasks
        # and actor setup, and rebuild workers that die — so one flaky/dead node
        # doesn't abort a long PB-scale map pipeline. cosmos owns FT for the
        # map-stage path; jude's own shuffle ops use Ray lineage + actor restarts.
        cfg = self.pipeline_config or self._default_cosmos_config()
        spec = jp.PipelineSpec(
            input_data=list(shards),
            stages=[jp.StageSpec(s) for s in self._stages],
            config=cfg,
        )
        out = jp.run_pipeline(spec)
        return list(out or [])

    @staticmethod
    def _default_cosmos_config() -> Any:
        """cosmos PipelineConfig with fault-tolerance defaults turned on (retries +
        worker rebuild). Only sets FT knobs the installed cosmos-xenna actually
        supports (older versions may lack some), falling back gracefully."""
        import os

        base: dict = {"execution_mode": jp.ExecutionMode.BATCH, "return_last_stage_outputs": True}
        ft = {
            "num_run_attempts_python": int(os.environ.get("JUDE_COSMOS_RUN_ATTEMPTS", "3")),
            "num_setup_attempts_python": int(os.environ.get("JUDE_COSMOS_SETUP_ATTEMPTS", "3")),
            "reset_workers_on_failure": True,
        }
        try:
            return jp.PipelineConfig(**base, **ft)
        except TypeError:
            # installed cosmos-xenna doesn't expose these knobs — keep the base.
            return jp.PipelineConfig(**base)

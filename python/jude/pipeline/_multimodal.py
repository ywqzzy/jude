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
        table), this streams the source's tasks with bounded memory
        (``jude.datasource.read_stream``) into the pipeline's input shards — so
        sources larger than memory (or a streaming video-frame source) feed the
        multi-stage pipeline without materializing the whole input at once.
        """
        import pyarrow as pa

        from jude import datasource

        p = cls(**kw)
        batches = list(datasource.read_stream(source, batch_rows=batch_rows))
        p._input_table = pa.Table.from_batches(batches) if batches else pa.Table.from_batches([], schema=source.schema())
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
        if self._input_table is None:
            raise ValueError("RelationPipeline has no source; use from_relation/from_table/from_source")
        return relation_to_shards(self._input_table, num_shards=self.num_shards, rows_per_shard=self.rows_per_shard)

    def _use_cosmos(self) -> bool:
        if self.engine == "cosmos":
            if not jp.is_cosmos_backed():
                raise RuntimeError("engine='cosmos' but cosmos-xenna is not installed")
            return True
        if self.engine == "auto":
            return jp.is_cosmos_backed()
        return False

    def run(self) -> pa.Table:
        """Execute the pipeline and return the sink as an Arrow table.

        Records into jude's observability registry (jude.observe) so multi-stage
        pipelines show up on the dashboard alongside SQL/UDF work: one query per
        run, one stage per pipeline stage, with row counts.
        """
        from jude import observe

        engine = "cosmos" if self._use_cosmos() else "local"
        shards = self._shards()
        if not self._stages:
            return shards_to_table(shards)
        with observe.query(f"pipeline[{engine}]: {len(self._stages)} stages", kind="pipeline") as q:
            for st in self._stages:
                q.stage(type(st).__name__).done()
            if self._use_cosmos():
                out = self._run_cosmos(shards)
            else:
                out = self._run_local(shards)
            table = shards_to_table(out)
            q.done(rows=table.num_rows, nbytes=table.nbytes)
            return table

    def to_relation(self, con: Any = None) -> Any:
        """Execute the pipeline and land the sink as a queryable jude Relation."""
        if con is None:
            import jude

            con = jude.connect()
        return con.from_arrow(self.run())

    def _run_local(self, shards: list[pa.Table]) -> list[pa.Table]:
        """Sequential in-process engine sharing the cosmos Stage API."""
        data = shards
        for stage in self._stages:
            stage.setup(None)
        try:
            for stage in self._stages:
                bs = max(1, int(getattr(stage, "stage_batch_size", 1)))
                out: list[pa.Table] = []
                for start in range(0, len(data), bs):
                    batch = data[start : start + bs]
                    res = stage.process_data(batch)
                    if res:
                        out.extend(res)
                data = out
        finally:
            for stage in self._stages:
                try:
                    stage.destroy()
                except Exception:
                    pass
        return data

    def _run_cosmos(self, shards: list[pa.Table]) -> list[pa.Table]:
        """Run on cosmos-xenna: each stage a StageSpec, shards as input_data."""
        for stage in self._stages:
            _pickle_by_value(stage)
        # A user-supplied PipelineConfig (e.g. raised monitoring_verbosity_level)
        # wins; otherwise jude's default finite-batch config.
        cfg = self.pipeline_config or jp.PipelineConfig(
            execution_mode=jp.ExecutionMode.BATCH,
            return_last_stage_outputs=True,
        )
        spec = jp.PipelineSpec(
            input_data=list(shards),
            stages=[jp.StageSpec(s) for s in self._stages],
            config=cfg,
        )
        out = jp.run_pipeline(spec)
        return list(out or [])
